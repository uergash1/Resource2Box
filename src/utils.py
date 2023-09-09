import argparse
import torch
from tqdm import tqdm
from sklearn.metrics import ndcg_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='clueweb09b_100', type=str, help='dataset')
    parser.add_argument("--title", default=True, type=bool, help='include document title')
    parser.add_argument("--body", default=True, type=bool, help='include document body')
    parser.add_argument("--train_pair_count", default=1000, type=int, help='number of train pairs')
    parser.add_argument("--train", default=80, type=int, help='number of train queries')
    parser.add_argument("--test", default=20, type=int, help='number of test queries')
    parser.add_argument("--folds", default=5, type=int, help='number of folds for cross validation')
    parser.add_argument("--random_seed", default=10, type=int, help='random seed')
    parser.add_argument("--gpu", default='0', type=str, help='gpu number')

    parser.add_argument("--ndcg_k", default=[2, 4, 6, 8, 10], type=list, help='nDCG results at k slice')
    parser.add_argument("--eval_test", default=True, type=bool, help='Evaluate test data in each epoch')
    parser.add_argument("--eval_train", default=True, type=bool, help='Evaluate train data in each epoch')

    parser.add_argument("--batch_size", default=128, type=int, help='batch size')
    parser.add_argument("--epochs", default=40, type=int, help='number of epochs')
    parser.add_argument("--learning_rate", default=1e-4, type=float, help='learning rate')
    parser.add_argument("--dim", default=768, type=int, help='embedding dimension')

    parser.add_argument("--box_type", default="geometric", type=str, help='box embedding type')
    parser.add_argument("--gamma", default=0.0, type=float, help='box-vector distance parameter')
    parser.add_argument("--delta", default=1.0, type=float, help='margin in hinge loss')

    return parser.parse_args()


def vector_box_distance(vector, center, offset, gamma=0.3):
    if vector.dim() == 1:
        vector = vector.unsqueeze(0)
        center = center.unsqueeze(0)
        offset = offset.unsqueeze(0)

    lower_left = center - offset
    upper_right = center + offset
    dist_out = torch.sum((torch.relu(vector - upper_right) + torch.relu(lower_left - vector)) ** 2, 1)
    dist_in = torch.sum((center - torch.min(upper_right, torch.max(lower_left, vector))) ** 2, 1)
    dist = dist_out + gamma * dist_in
    return dist


def ranking_loss(query_point, pos_center, pos_offset, neg_center, neg_offset, delta):
    pos_distance = vector_box_distance(query_point, pos_center, pos_offset)
    neg_distance = vector_box_distance(query_point, neg_center, neg_offset)
    loss = torch.relu(pos_distance - neg_distance + delta)
    return loss.mean()


def ndcg_eval(model, query_layer, data, current_fold, mode, k, device):
    ndcg_results = {}
    model.eval()

    with torch.no_grad():
        query_embeddings, document_embeddings, y_true = data.get_eval_data(current_fold, mode)
        query_embeddings = query_embeddings.to(device)
        document_embeddings = document_embeddings.to(device)

        center, offset = model(document_embeddings)

        # Compute distance between test query and data source boxes
        y_score = []
        for query_embedding in tqdm(query_embeddings):
            resource_y_score = []
            for resource_id in range(center.shape[0]):
                dist = vector_box_distance(query_layer(query_embedding), center[resource_id], offset[resource_id])
                resource_y_score.append(- dist.item())
            y_score.append(resource_y_score)

        for kk in k:
            ndcg_results[f"nDCG @{kk}"] = ndcg_score(y_true, y_score, k=kk)

    return ndcg_results