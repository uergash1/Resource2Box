from transformers import BertModel, BertTokenizer
import torch
import pandas as pd
import random


class Dataset:
    def __init__(self, args, device):
        super(Dataset, self).__init__()

        self.args = args
        self.device = device
        self.dataset_name = args.dataset
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = self.get_bert_model()
        self.__load_query_resource_artifacts__()

    def __load_query_resource_artifacts__(self):
        self.resource_query_similarity = pd.read_csv(f'../data/{self.dataset_name}/processed/resource_query_similarity.tsv', sep='\t')
        self.rname_to_id = dict([(name, id) for id, name in enumerate(sorted(self.resource_query_similarity['resource_id'].unique().tolist()))])
        self.id_to_rname = dict([(id, name) for id, name in enumerate(sorted(self.resource_query_similarity['resource_id'].unique().tolist()))])
        self.qname_to_id = dict([(name, id) for id, name in enumerate(sorted(self.resource_query_similarity['query_id'].unique().tolist()))])
        self.id_to_qname = dict([(id, name) for id, name in enumerate(sorted(self.resource_query_similarity['query_id'].unique().tolist()))])

        # Loading documents
        self.documents = {}
        for resource_id in self.rname_to_id.keys():
            rdata = pd.read_csv(f"../data/{self.dataset_name}/processed/resources/{resource_id}.tsv", sep='\t')

            if self.args.title and self.args.body:
                rdata['content'] = rdata['title'] + '\n\n' + rdata['body']
            elif self.args.title:
                rdata['content'] = rdata['title']
            else:
                rdata['content'] = rdata['body']
            self.documents[resource_id] = rdata['content'].tolist()

        # Loading queries
        qdata = pd.read_csv(f"../data/{self.dataset_name}/processed/queries.tsv", sep='\t')
        self.queries = dict(zip(qdata['id'], qdata['query']))

    def get_bert_model(self):
        bert = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        for param in bert.parameters():
            param.requires_grad = False
        return bert

    def get_bert_embedding(self, input):
        input_tokens = self.tokenizer(input, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.bert(**input_tokens)
        return output.last_hidden_state.mean(dim=1)

    def get_document_embedding(self, idx):
        document_embedding_list = []
        for i in idx:
            document_embeddings = self.get_bert_embedding(self.documents[self.id_to_rname[i.item()]])
            document_embedding_list.append(document_embeddings)
        return torch.stack(document_embedding_list)

    def get_batch_embeddings(self, query_idx_batch, pos_idx_batch, neg_idx_batch):
        actual_queries = [self.queries[self.id_to_qname[i.item()]] for i in query_idx_batch]
        query_embedding_batch = self.get_bert_embedding(actual_queries)
        pos_resource_embedding_batch = self.get_document_embedding(pos_idx_batch)
        neg_resource_embedding_batch = self.get_document_embedding(neg_idx_batch)
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
        actual_queries = [self.queries[self.id_to_qname[i.item()]] for i in query_portion]
        query_embeddings = self.get_bert_embedding(actual_queries)
        document_embeddings = self.get_document_embedding(list(self.id_to_rname.keys()))
        for query_id in query_portion:
            query_y_true = self.resource_query_similarity[self.resource_query_similarity['query_id'] == self.id_to_qname[query_id]]['similarity_score'].tolist()
            y_true.append(query_y_true)
        return query_embeddings, document_embeddings, y_true


    def get_train_pairs(self, current_fold):
        train_pairs = []
        current_fold_queries = self.get_train_test_portion(current_fold, mode='train')
        for count in range(self.args.train_pair_count):
            query_id = random.choice(current_fold_queries)
            query_name = self.id_to_qname[query_id]
            pos_resources = self.resource_query_similarity[(self.resource_query_similarity['query_id'] == query_name) & (self.resource_query_similarity['similarity_score'] > 0.0)]['resource_id'].tolist()
            pos_resource = random.choice(pos_resources)

            neg_resources = self.resource_query_similarity[
                (self.resource_query_similarity['query_id'] == query_name) & (
                            self.resource_query_similarity['similarity_score'] <= 0.0)]['resource_id'].tolist()
            neg_resource = random.choice(neg_resources)

            train_pairs.append((query_id, self.rname_to_id[pos_resource], self.rname_to_id[neg_resource]))
        return torch.tensor(train_pairs, dtype=torch.long)
