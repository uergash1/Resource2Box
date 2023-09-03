import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, embedding_dim):
        super(Attention, self).__init__()
        self.linear = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        # x shape: [num_documents, embedding_dim]
        weights = torch.nn.functional.softmax(self.linear(x).squeeze(-1), dim=0)
        # weights shape after unsqueeze and multiplication: [num_documents, 1] * [num_documents, embedding_dim]
        weighted_sum = (weights.unsqueeze(-1) * x).sum(dim=0)
        # weighted_sum shape: [embedding_dim]
        return weighted_sum.unsqueeze(0)  # shape: [1, embedding_dim]
