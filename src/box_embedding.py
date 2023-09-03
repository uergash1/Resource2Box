import torch.nn as nn
import torch


class BoxEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(BoxEmbedding, self).__init__()
        self.center_layer = nn.Linear(embedding_dim, embedding_dim)
        self.scale_layer = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        center = self.center_layer(x)
        offset = torch.exp(self.scale_layer(x))
        return center, offset
