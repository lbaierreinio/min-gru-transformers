import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.utility import parallel_scan_log

class ParallelMinGRU(nn.Module):
    
    def __init__(self, dim_x, dim_h): # Note: MinGRU paper suggests embedding dimension of 128
        super().__init__()
        self.linear_z = nn.Linear(dim_x, dim_h) # Linear layer for producing z from x
        self.linear_h = nn.Linear(dim_x, dim_h) # Linear layer for producing candidate state h_tilde from x
        self.linear_o = nn.Linear(dim_h, dim_x) # Linear layer for producing output from hidden state

    def log_g(self, x):
        return torch.where(x >= 0, torch.log(F.relu(x)+0.5), -F.softplus(-x))

    def forward(self, x, h_prev):
        """
        Compute the forward pass.

        Args:
            x: torch.Tensor, shape (batch_size, seq_len, input_size)
            h_prev: torch.Tensor, shape (batch_size, 1, hidden_size)
        
        Returns:
            h: torch.Tensor, shape (batch_size, seq_len, input_size) # TODO: Verify this is correct output dimension
        """
        k = self.linear_z(x) 
        tilde_h = self.linear_h(x) # Candidate state
        log_z = -F.softplus(-k) # Log (z) 
        log_one_minus_z = -F.softplus(k) # Log (1 - z)
        log_h_prev = self.log_g(h_prev) # Previous hidden state
        log_tilde_h = self.log_g(tilde_h) # Log candidate state
        # Parallel scan (log z + log h_tilde since we are using log, and had z * h_tilde in the original implementation)
        h = parallel_scan_log(log_one_minus_z, torch.cat([log_h_prev, log_z + log_tilde_h], dim=1)) # parallel_scan returns h[1:t]
        out = self.linear_o(h)
        return out