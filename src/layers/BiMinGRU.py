import torch
from torch import nn
from layers.MinGRU import MinGRU

class BiMinGRU(nn.Module):
  def __init__(self, dim_in, dim_hidden):
    super().__init__()
    self.forward_rnn = MinGRU(dim_in, dim_hidden)
    self.backward_rnn = MinGRU(dim_in, dim_hidden)
    self.linear = nn.Linear(2 * dim_hidden, dim_hidden)
  
  def forward(self, x):
    x_reversed = x.flip(dims=[1])
    out_forward = self.forward_rnn(x)
    out_backward = self.backward_rnn(x_reversed)
    concat = torch.cat((out_forward, out_backward), dim=2)
    out = self.linear(concat)

    return out