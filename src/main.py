import torch
from attention import Attention
from box_embedding import BoxEmbedding
import torch.optim as optim
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import random
import warnings
import numpy as np
import utils
from data import Dataset

warnings.filterwarnings('ignore')
args = utils.parse_args()

random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)


def ranking_loss(query_point, pos_center, pos_offset, neg_center, neg_offset):
    pos_distance = torch.norm(query_point - pos_center, dim=-1) - pos_offset
    neg_distance = torch.norm(query_point - neg_center, dim=-1) - neg_offset
    loss = torch.relu(pos_distance - neg_distance + 1.0)  # Margin = 1.0
    return loss.mean()


########## GPU Settings ##########
if torch.cuda.is_available():
    device = torch.device("cuda:" + args.gpu)
else:
    device = torch.device("cpu")
print('Device:\t', device, '\n')


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased').to(device)

for param in bert.parameters():
    param.requires_grad = False

embedding_dim = args.dim  # BERT's output dimension

attention_layer = Attention(embedding_dim).to(device)
box_embedding = BoxEmbedding(embedding_dim).to(device)

params = list(attention_layer.parameters()) + list(box_embedding.parameters())
optimizer = optim.Adam(params, lr=args.learning_rate)

data = Dataset(args)

# Dummy data and training pairs
documents = data.load_documents()
queries = data.load_queries()
train_pairs = data.get_train_pairs()

for epoch in range(args.epochs):
    epoch_loss = 0.0
    for query_idx, pos_idx, neg_idx in tqdm(train_pairs, desc=f"Epoch {epoch+1}"):
        optimizer.zero_grad()

        # Process query
        query_tokens = tokenizer(queries[query_idx], return_tensors="pt", padding=True, truncation=True)
        query_output = bert(**query_tokens.to(device))
        query_embedding = query_output.last_hidden_state.mean(dim=1).to(device)

        # Process positive data source
        pos_docs = documents[pos_idx]
        pos_tokens = tokenizer(pos_docs, return_tensors="pt", padding=True, truncation=True)
        pos_output = bert(**pos_tokens.to(device))
        pos_doc_embeddings = pos_output.last_hidden_state.mean(dim=1)
        pos_data_source_embedding = attention_layer(pos_doc_embeddings.to(device))

        # Process negative data source
        neg_docs = documents[neg_idx]
        neg_tokens = tokenizer(neg_docs, return_tensors="pt", padding=True, truncation=True)
        neg_output = bert(**neg_tokens.to(device))
        neg_doc_embeddings = neg_output.last_hidden_state.mean(dim=1)
        neg_data_source_embedding = attention_layer(neg_doc_embeddings.to(device))

        # Get box embeddings
        pos_center, pos_offset = box_embedding(pos_data_source_embedding)
        neg_center, neg_offset = box_embedding(neg_data_source_embedding)

        # Compute loss
        loss = ranking_loss(query_embedding, pos_center, pos_offset, neg_center, neg_offset)
        epoch_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Average Loss: {epoch_loss / len(train_pairs)}")


# Evaluation
# def rank_data_sources(query_embedding, box_embedding, attention_layer):
#     centers, offsets = box_embedding(torch.arange(num_data_sources))
#     distances = torch.norm(query_embedding - centers, dim=-1) - offsets.squeeze()
#     ranking = torch.argsort(distances).tolist()
#     return ranking
#
# # Evaluate for a new query
# new_query = "This is a new query for evaluation"
# new_query_tokens = tokenizer(new_query, return_tensors="pt", padding=True, truncation=True)
# new_query_output = bert(**new_query_tokens)
# new_query_embedding = new_query_output.last_hidden_state.mean(dim=1)
#
# ranking = rank_data_sources(new_query_embedding, box_embedding)
# print("Ranked data sources:", ranking)

