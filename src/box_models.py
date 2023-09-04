import torch
import torch.nn as nn


class GeometricBox(nn.Module):
    def __init__(self, embedding_dim):
        super(GeometricBox, self).__init__()
        self.dim = embedding_dim
        self.W_center = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_offset = nn.Linear(embedding_dim, embedding_dim, bias=True)

    def forward(self, x):
        center = self.W_center(x)
        offset = torch.relu(self.W_offset(x))
        return center, offset