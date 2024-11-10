import torch
import torch.nn as nn

class SequentialMinGRU(nn.Module):
    def __init__(self, dim_x, dim_h):
        super().__init__()
        self.linear_z = nn.Linear(dim_x, dim_h)
        self.linear_h = nn.Linear(dim_x, dim_h)
    
    '''
    x_t: (batch_size, input_size)
    h_prev: (batch_size, hidden size)
    '''
    def forward(self, x_t, h_prev):
        z_t = torch.sigmoid(self.linear_z(x_t)) # Output gate
        h_tilde = self.linear_h(x_t) # Candidate cell
        h_t = ((1 - z_t) * h_prev) + (z_t * h_tilde) # New hidden state
        return h_t

