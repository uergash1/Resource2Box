import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, embedding_dim):
        super(Attention, self).__init__()
        self.embedding_dim = embedding_dim
        self.query = nn.Parameter(torch.randn(embedding_dim))
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        # x: [num_docs, embedding_dim]

        # Compute keys and values
        keys = self.key(x)  # [num_docs, embedding_dim]
        values = self.value(x)  # [num_docs, embedding_dim]

        # Compute attention scores
        query = self.query.unsqueeze(0)  # [1, embedding_dim]
        scores = torch.matmul(keys, query.transpose(-2, -1))  # [num_docs, 1]
        scores = scores.squeeze(-1)  # [num_docs]
        attention_weights = F.softmax(scores, dim=-1)  # [num_docs]

        # Compute aggregated embeddings
        aggregated_embedding = torch.sum(values * attention_weights.unsqueeze(-1), dim=0)  # [embedding_dim]

        return aggregated_embedding


# class Attention(nn.Module):
#     def __init__(self, embedding_dim):
#         super(Attention, self).__init__()
#         self.linear = nn.Linear(embedding_dim, 1)
#
#     def forward(self, x):
#         # x shape: [num_documents, embedding_dim]
#         weights = torch.nn.functional.softmax(self.linear(x).squeeze(-1), dim=0)
#         # weights shape after unsqueeze and multiplication: [num_documents, 1] * [num_documents, embedding_dim]
#         weighted_sum = (weights.unsqueeze(-1) * x).sum(dim=0)
#         # weighted_sum shape: [embedding_dim]
#         return weighted_sum.unsqueeze(0)  # shape: [1, embedding_dim]
