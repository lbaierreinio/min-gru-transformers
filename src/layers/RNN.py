import torch.nn as nn
from layers.MinGRU import MinGRU
from layers.BiMinGRU import BiMinGRU

class RNN(nn.Module):
  def __init__(self, *, embedding_dim, inner_dim, num_layers=1, bidirectional=False):
    super().__init__()

    self.linear_in = nn.Linear(embedding_dim, inner_dim)

    self.layers = nn.ModuleList([
        (BiMinGRU(inner_dim) if bidirectional else MinGRU(inner_dim)) for _ in range(num_layers)
    ])

  def forward(self, x):
    x = self.linear_in(x)
    for layer in self.layers:
      x = layer(x)

    return x