from attention import Attention
from box_models import *


class QueryModel(nn.Module):
    def __init__(self, hidden_dim, bert_dim=768):
        super(QueryModel, self).__init__()
        self.layer = nn.Linear(bert_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.layer(x)


class Model(nn.Module):
    def __init__(self, box_type, hidden_dim, bert_dim=768):
        super(Model, self).__init__()
        # Geometric or Attentive
        self.box_type = box_type
        self.attention = Attention(bert_dim, hidden_dim)
        if self.box_type == 'geometric':
            self.box = GeometricBox(hidden_dim)

    def forward(self, x):
        x = self.attention(x)
        return self.box(x)

