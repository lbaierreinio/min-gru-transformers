import torch
from torch import nn
from layers.rnn.MinGRU import MinGRU


class BiMinGRU(nn.Module):
    def __init__(self, dim_in, dim_hidden):
        super().__init__()
        self.forward_rnn = MinGRU(dim_in, dim_hidden)
        self.backward_rnn = MinGRU(dim_in, dim_hidden)
        self.linear = nn.Linear(2 * dim_hidden, dim_hidden)

    def forward(self, x, mask=None, h_prev=None):
        if h_prev is not None: 
            h_prev_forward, h_prev_backward = h_prev # (B, H)
            x = x # (B, E)
            mask = mask # (B)
            h_forward = self.forward_rnn(x, mask=mask, h_prev=h_prev_forward)
            h_backward = self.backward_rnn(x, mask=mask, h_prev=h_prev_backward)
            concat = torch.cat((h_forward, h_backward), dim=1)
            out = self.linear(concat)
            return out, h_forward, h_backward
   
        else:
            x_reversed = x.flip(dims=[1])
            mask_reversed = mask.flip(dims=[1]) if mask is not None else None
            out_forward = self.forward_rnn(x, mask=mask)
            out_backward = self.backward_rnn(x_reversed, mask=mask_reversed)
            concat = torch.cat((out_forward, out_backward.flip(dims=[1])), dim=2)
            out = self.linear(concat)

            return out, None, None
