import torch.nn as nn
import torch
from attention import Attention
from box_models import *


class Model(nn.Module):
    def __init__(self, box_type, embedding_dim):
        super(Model, self).__init__()
        # Geometric or Attentive
        self.box_type = box_type
        self.dim = embedding_dim
        self.attention = Attention(embedding_dim)
        if self.box_type == 'geometric':
            self.box = GeometricBox(embedding_dim)
        # if self.box_type == 'geometric':
        #     self.box = GeometricBox(embedding_dim)
        #
        # elif self.box_type == 'attentive':
        #     self.box = AttentiveBox(embedding_dim)

    def forward(self, x):
        x = self.attention(x)
        return self.box(x)

