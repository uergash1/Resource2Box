import torch.nn as nn
import torch


class BoxEmbedding(nn.Module):
    def __init__(self, box_type, embedding_dim):
        super(BoxEmbedding, self).__init__()
        # Geometric or Attentive
        self.box_type = box_type
        self.dim = embedding_dim

        if self.box_type == 'geometric':
            self.W_center = nn.Linear(embedding_dim, embedding_dim, bias=False)
            self.W_offset = nn.Linear(embedding_dim, embedding_dim, bias=False)

        elif self.box_type == 'attentive':
            self.W_key = nn.Linear(embedding_dim, embedding_dim, bias=False)
            self.W_val = nn.Linear(embedding_dim, embedding_dim, bias=False)
            self.query = nn.Parameter(torch.randn(embedding_dim, 1))
            self.W_center = nn.Linear(embedding_dim, embedding_dim, bias=False)
            self.W_offset = nn.Linear(embedding_dim, embedding_dim, bias=True)

    def forward(self, x):
        if self.box_type == 'geometric':
            min_point = torch.min(x, 0)[0]
            max_point = torch.max(x, 0)[0]
            center = self.W_center((max_point + min_point) / 2)
            offset = self.W_offset((max_point - min_point) / 2)

        elif self.box_type == 'attentive':
            key = self.W_key(x)
            value = self.W_val(x)
            attention = torch.softmax(torch.matmul(key, self.query) / (self.dim ** 0.5), 0)
            aggregation = torch.matmul(value.T, attention).squeeze(1)
            center = self.W_center(aggregation)
            offset = torch.relu(self.W_offset(aggregation))

        return center, offset
