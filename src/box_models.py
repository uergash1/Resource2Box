import torch
import torch.nn as nn


class GeometricBox(nn.Module):
    def __init__(self, embedding_dim):
        super(GeometricBox, self).__init__()
        self.W_center = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_offset = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x):
        batch_size, num_document, dim = x.shape

        min_point = torch.min(x, 1)[0]
        max_point = torch.max(x, 1)[0]
        center = self.W_center((max_point + min_point) / 2)
        offset = self.W_offset((max_point - min_point) / 2)
        return center, offset


class AttentiveBox(nn.Module):
    def __init__(self, embedding_dim):
        super(AttentiveBox, self).__init__()
        self.dim = embedding_dim
        self.W_key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_val = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.query = nn.Parameter(torch.randn(embedding_dim, 1))
        self.W_center = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_offset = nn.Linear(embedding_dim, embedding_dim, bias=True)

    def forward(self, x):
        batch_size, num_document, dim = x.shape

        key = self.W_key(x)
        value = self.W_val(x)
        query = self.query.unsqueeze(0).repeat(batch_size, 1, 1)

        attention = torch.softmax(torch.bmm(key, query) / (self.dim ** 0.5), 1)
        aggregation = torch.sum(attention * value, 1)

        center = self.W_center(aggregation)
        offset = torch.relu(self.W_offset(aggregation))
        return center, offset