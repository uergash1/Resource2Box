import torch.nn as nn
import torch
from box_models import *


class BoxEmbedding(nn.Module):
    def __init__(self, box_type, embedding_dim):
        super(BoxEmbedding, self).__init__()
        # Geometric or Attentive
        self.box_type = box_type
        self.dim = embedding_dim

        if self.box_type == 'geometric':
            self.box = GeometricBox(embedding_dim)

        elif self.box_type == 'attentive':
            self.box = AttentiveBox(embedding_dim)

    def forward(self, x):
        return self.box(x)

