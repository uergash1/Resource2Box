import torch
import numpy as np
import pandas as pd
import pickle as pkl
import os
import random
from tqdm import tqdm, trange
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.utils import to_undirected

class Dataset:
    def __init__(self, args, device):
        super(Dataset, self).__init__()

        self.args = args
        self.device = device
        self.dataset_name = args.dataset
        self.__load_query_resource_artifacts__()

    def __load_query_resource_artifacts__(self):
        self.resource_query_similarity = pd.read_csv(
            f'../data/{self.dataset_name}/processed/resource_query_similarity.tsv', sep='\t')
        self.rname_to_id = dict([(name, id) for id, name in
                                 enumerate(sorted(self.resource_query_similarity['resource_id'].unique().tolist()))])
        self.id_to_rname = dict([(id, name) for id, name in
                                 enumerate(sorted(self.resource_query_similarity['resource_id'].unique().tolist()))])
        self.qname_to_id = dict([(name, id) for id, name in
                                 enumerate(sorted(self.resource_query_similarity['query_id'].unique().tolist()))])
        self.id_to_qname = dict([(id, name) for id, name in
                                 enumerate(sorted(self.resource_query_similarity['query_id'].unique().tolist()))])

        # Loading documents
        self.documents = {}
        for resource_id in self.rname_to_id.keys():
            self.documents[resource_id] = np.load(f"../data/{self.dataset_name}/embeddings/resources/title_body/{resource_id}.npy")

        # Loading queries
        self.queries = np.load(f"../data/{self.dataset_name}/embeddings/queries.npy")

    def construct_resource_graph(self):
        graph_file = f'graphs/{self.args.dataset}_threshold{self.args.threshold}.npy'

        if not os.path.exists(graph_file):
            self.edge_index, self.edge_weight = [], []
            for i in trange(len(self.id_to_rname)):
                resource_i = self.id_to_rname[i]
                for j in range(i + 1, len(self.id_to_rname)):
                    resource_j = self.id_to_rname[j]
                    sim = cosine_similarity(self.documents[resource_i], self.documents[resource_j])
                    sim_ij = np.sum(sim > self.args.threshold) / (sim.shape[0] * sim.shape[1])
                    if sim_ij > 0:
                        self.edge_index.append([i, j])
                        self.edge_weight.append(sim_ij)

            self.edge_index = torch.tensor(self.edge_index).T
            self.edge_weight = torch.tensor(self.edge_weight)
            self.edge_index, self.edge_weight = to_undirected(self.edge_index, self.edge_weight)

            with open(graph_file, 'wb') as f:
                pkl.dump([self.edge_index, self.edge_weight], f)

        else:
            with open(graph_file, 'rb') as f:
                self.edge_index, self.edge_weight = pkl.load(f)

    def get_batch_embeddings(self, query_idx_batch, pos_idx_batch, neg_idx_batch):
        query_embedding_batch = torch.Tensor([list(self.queries[i]) for i in query_idx_batch])
        pos_resource_embedding_batch = torch.Tensor([list(self.documents[self.id_to_rname[i]]) for i in pos_idx_batch])
        neg_resource_embedding_batch = torch.Tensor([list(self.documents[self.id_to_rname[i]]) for i in neg_idx_batch])
        return query_embedding_batch, pos_resource_embedding_batch, neg_resource_embedding_batch

    def get_train_test_portion(self, current_fold, mode):
        query_ids = list(self.id_to_qname.keys())
        portion = []
        if mode == 'test':
            portion = query_ids[self.args.test * current_fold:self.args.test * (current_fold + 1)]
        elif mode == 'train':
            portion = query_ids[:self.args.test * current_fold] + query_ids[self.args.test * (current_fold + 1):]
        return portion

    def get_eval_data(self, current_fold, mode):
        y_true = []
        query_portion = self.get_train_test_portion(current_fold, mode)
        query_embeddings = torch.Tensor([self.queries[i] for i in query_portion])
        document_embeddings = torch.Tensor(list(self.documents.values()))
        for query_id in query_portion:
            query_y_true = \
            self.resource_query_similarity[self.resource_query_similarity['query_id'] == self.id_to_qname[query_id]][
                'similarity_score'].tolist()
            y_true.append(query_y_true)
        return query_embeddings, document_embeddings, y_true

    def get_train_pairs(self, current_fold):
        train_pairs = []
        current_fold_queries = self.get_train_test_portion(current_fold, mode='train')
        for count in range(self.args.train_pair_count):
            query_id = random.choice(current_fold_queries)
            query_name = self.id_to_qname[query_id]
            pos_resources = self.resource_query_similarity[
                (self.resource_query_similarity['query_id'] == query_name) & (
                            self.resource_query_similarity['similarity_score'] > 0.0)]['resource_id'].tolist()
            pos_resource = random.choice(pos_resources)

            pos_resource_score = self.resource_query_similarity[(self.resource_query_similarity['query_id'] == query_name)
                                                                & (self.resource_query_similarity['resource_id'] == pos_resource)]['similarity_score'].values[0]

            neg_resources = self.resource_query_similarity[
                (self.resource_query_similarity['query_id'] == query_name) & (
                        self.resource_query_similarity['similarity_score'] < pos_resource_score)]['resource_id'].tolist()
            neg_resource = random.choice(neg_resources)

            train_pairs.append((query_id, self.rname_to_id[pos_resource], self.rname_to_id[neg_resource]))
        return torch.tensor(train_pairs, dtype=torch.long)