import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.utility import parallel_scan_log

class ParallelMinGRU(nn.Module):
    def __init__(self, dim_x, dim_h):
        super().__init__()
        self.linear_z = nn.Linear(dim_x, dim_h)
        self.linear_h = nn.Linear(dim_x, dim_h)
    
    '''
    x: (batch_size, seq_len, input_size)
    h_0: (batch_size, 1, hidden_size)
    '''
    def forward(self, x, h_0):
        z = torch.sigmoid(self.linear_z(x))
        h_tilde = self.linear_h(x)
        h = parallel_scan_log((1-z), torch.cat([h_0, z*h_tilde], dim=1))
        return h


