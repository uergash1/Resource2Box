import torch
from torch.utils.data import DataLoader, TensorDataset
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


def vector_box_distance(vector, center, offset, gamma=0.3):
    lower_left = center - offset
    upper_right = center + offset
    dist_out = torch.sum((torch.relu(vector - upper_right) + torch.relu(lower_left - vector)) ** 2)
    dist_in = torch.sum((center - torch.min(upper_right, torch.max(lower_left, vector))) ** 2)
    dist = dist_out + gamma * dist_in
    return dist


def ranking_loss(query_point, pos_center, pos_offset, neg_center, neg_offset):
    pos_distance = vector_box_distance(query_point, pos_center, pos_offset)
    neg_distance = vector_box_distance(query_point, neg_center, neg_offset)
    loss = torch.relu(pos_distance - neg_distance + 1.0)  # Margin = 1.0
    return loss.mean()


########## GPU Settings ##########
if torch.cuda.is_available():
    device = torch.device("cuda:" + args.gpu)
else:
    device = torch.device("cpu")
print('Device:\t', device, '\n')

box_embedding = BoxEmbedding(args.box_type, args.dim).to(device)

params = list(box_embedding.parameters())
optimizer = optim.Adam(params, lr=args.learning_rate)

data = Dataset(args, device)
# documents = data.load_documents()
# queries = data.load_queries()
train_pairs = data.get_train_pairs()
# Create a DataLoader
train_data = TensorDataset(train_pairs)
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
# We should change this later.
# test_pairs = data.get_train_pairs()

for epoch in range(args.epochs):
    epoch_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
        optimizer.zero_grad()

        # Unpack the batch
        batch_data = batch[0]
        query_idx_batch, pos_idx_batch, neg_idx_batch = batch_data[:, 0], batch_data[:, 1], batch_data[:, 2]

        query_embedding_batch, pos_resource_embedding_batch, neg_resource_embedding_batch = data.get_batch_embeddings(query_idx_batch, pos_idx_batch, neg_idx_batch)

        # # Process query
        # query_tokens = tokenizer(queries[query_idx], return_tensors="pt", padding=True, truncation=True)
        # query_output = bert(**query_tokens.to(device))
        # query_embedding = query_output.last_hidden_state.mean(dim=1).squeeze(0).to(device)
        #
        # # Process positive data source
        # pos_docs = documents[pos_idx]
        # pos_tokens = tokenizer(pos_docs, return_tensors="pt", padding=True, truncation=True)
        # pos_output = bert(**pos_tokens.to(device))
        # pos_doc_embeddings = pos_output.last_hidden_state.mean(dim=1)
        #
        # # Process negative data source
        # neg_docs = documents[neg_idx]
        # neg_tokens = tokenizer(neg_docs, return_tensors="pt", padding=True, truncation=True)
        # neg_output = bert(**neg_tokens.to(device))
        # neg_doc_embeddings = neg_output.last_hidden_state.mean(dim=1)

        # Get box embeddings
        pos_center, pos_offset = box_embedding(pos_resource_embedding_batch)
        neg_center, neg_offset = box_embedding(neg_resource_embedding_batch)

        # Compute loss
        loss = ranking_loss(query_embedding_batch, pos_center, pos_offset, neg_center, neg_offset)
        epoch_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

        if args.box_type == 'geometric':
            box_embedding.W_offset.weight.data = box_embedding.W_offset.weight.data.clamp(min=1e-6)

    print(f"Epoch {epoch + 1}, Average Loss: {epoch_loss / len(train_pairs)}")

# box_embedding.eval()
#
# with torch.no_grad():
#     # Get box embeddings of each data source.
#     data_source_box = {}
#     for idx in tqdm(documents):
#         docs = documents[idx]
#         tokens = tokenizer(docs, return_tensors="pt", padding=True, truncation=True)
#         output = bert(**tokens.to(device))
#         doc_embeddings = output.last_hidden_state.mean(dim=1)
#         center, offset = box_embedding(doc_embeddings)
#         data_source_box[idx] = (center, offset)
#
#     # Compute distance between test query and data source boxes
#     for query_idx, pos_idx, neg_idx in tqdm(test_pairs):
#         query_tokens = tokenizer(queries[query_idx], return_tensors="pt", padding=True, truncation=True)
#         query_output = bert(**query_tokens.to(device))
#         query_embedding = query_output.last_hidden_state.mean(dim=1).to(device)
#
#         data_source_dist = {}
#         for idx in data_source_box:
#             center, offset = data_source_box[idx]
#             dist = torch.mean(torch.norm(query_embedding - center, dim=-1) - offset)
#             data_source_dist[idx] = dist
#
#         data_source_rank = sorted(data_source_dist, key=lambda x: data_source_dist[x])

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

