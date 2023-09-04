import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import ndcg_score
from box_embedding import Model
import torch.optim as optim
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

########## GPU Settings ##########
if torch.cuda.is_available():
    device = torch.device("cuda:" + args.gpu)
else:
    device = torch.device("cpu")
print('Device:\t', device, '\n')


def ndcg_eval(model, data, current_fold, mode, k):
    ndcg_results = {}
    model.eval()

    with torch.no_grad():
        query_embeddings, document_embeddings, y_true = data.get_eval_data(current_fold, mode)
        center, offset = model(document_embeddings)

        # Compute distance between test query and data source boxes
        y_score = []
        for query_embedding in tqdm(query_embeddings):
            resource_y_score = []
            for resource_id in range(center.shape[0]):
                dist = torch.mean(torch.norm(query_embedding - center[resource_id], dim=-1) - offset[resource_id])
                resource_y_score.append(dist.item())
            y_score.append(resource_y_score)

        for kk in k:
            ndcg_results[kk] = ndcg_score(y_true, y_score, k=kk)
    return ndcg_results


def train(model, data, current_fold):
    params = list(model.parameters())
    optimizer = optim.Adam(params, lr=args.learning_rate)

    train_pairs = data.get_train_pairs(current_fold)
    train_data = TensorDataset(train_pairs)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            optimizer.zero_grad()

            batch_data = batch[0]
            query_idx_batch, pos_idx_batch, neg_idx_batch = batch_data[:, 0], batch_data[:, 1], batch_data[:, 2]

            query_embedding_batch, pos_resource_batch, neg_resource_batch = data.get_batch_embeddings(query_idx_batch,
                                                                                                      pos_idx_batch,
                                                                                                      neg_idx_batch)

            pos_center, pos_offset = model(pos_resource_batch)
            neg_center, neg_offset = model(neg_resource_batch)

            loss = utils.ranking_loss(query_embedding_batch, pos_center, pos_offset, neg_center, neg_offset)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

            if args.box_type == 'geometric':
                model.W_offset.weight.data = model.W_offset.weight.data.clamp(min=1e-6)

        print(f"Epoch {epoch + 1}, Average Loss: {epoch_loss / len(train_pairs)}")

        if args.eval_train:
            ndcg_results = ndcg_eval(model, data, current_fold, mode='train', k=args.ndcg_k)
            print(f"Train data nDCG results: {ndcg_results}")
        if args.eval_test:
            ndcg_results = ndcg_eval(model, data, current_fold, mode='test', k=args.ndcg_k)
            print(f"Test data nDCG results: {ndcg_results}")



def main():
    data = Dataset(args, device)

    # Train n number of folds for cross validation
    for current_fold in range(args.folds):
        model = Model(args.box_type, args.dim).to(device)
        train(model, data, current_fold)
        ndcg_eval(model, data, current_fold, mode='test', k=args.ndcg_k)


if __name__ == "__main__":
    main()