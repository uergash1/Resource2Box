import pandas as pd
import random


class Dataset:
    def __init__(self, args):
        super(Dataset, self).__init__()

        self.args = args
        self.dataset_name = args.dataset
        self.resource_query_similarity = pd.read_csv(f'../data/{self.dataset_name}/processed/resource_query_similarity.tsv', sep='\t')
        self.resources = sorted(self.resource_query_similarity['resource_id'].unique().tolist())
        self.queries = sorted(self.resource_query_similarity['query_id'].unique().tolist())

    def load_documents(self):
        documents = {}
        for resource_id in self.resources:
            data = pd.read_csv(f"../data/fedweb14/processed/resources/{resource_id}.tsv", sep='\t')

            if self.args.title and self.args.body:
                data['content'] = data['title'] + '\n\n' + data['body']
            elif self.args.title:
                data['content'] = data['title']
            else:
                data['content'] = data['body']
            documents[resource_id] = data['content'].tolist()
        return documents

    def load_queries(self):
        data = pd.read_csv("../data/fedweb14/processed/queries.tsv", sep='\t')
        queries = dict(zip(data['id'], data['query']))
        return queries

    def get_train_pairs(self):
        train_pairs = []
        for count in range(self.args.train_pair_count):
            query = random.choice(self.queries[:self.args.train])
            pos_resources = self.resource_query_similarity[(self.resource_query_similarity['query_id'] == query) & (self.resource_query_similarity['similarity_score'] > 0.0)]['resource_id'].tolist()
            pos_resource = random.choice(pos_resources)

            neg_resources = self.resource_query_similarity[
                (self.resource_query_similarity['query_id'] == query) & (
                            self.resource_query_similarity['similarity_score'] <= 0.0)]['resource_id'].tolist()
            neg_resource = random.choice(neg_resources)

            train_pairs.append((query, pos_resource, neg_resource))
        return train_pairs
