import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.utility import parallel_scan_log

class ParallelMinGRU(nn.Module):
    def __init__(self, dim_x, dim_h):
        super().__init__()
        self.linear_z = nn.Linear(dim_x, dim_h) # Linear layer for producing z from x
        self.linear_h = nn.Linear(dim_x, dim_h) # Linear layer for producing candidate state h_tilde from x

    def log_g(self, x):
        return torch.where(x >= 0, torch.log(F.relu(x)+0.5), -F.softplus(-x))

    def forward(self, x, h_0): # TODO: Figure out what h_0 is
        """
        Compute the forward pass.

        Args:
            x: torch.Tensor, shape (batch_size, seq_len, input_size)
            h_0: torch.Tensor, shape (batch_size, 1, hidden_size)
        
        Returns:
            h: torch.Tensor, shape (batch_size, seq_len, hidden_size)
        """
        k = self.linear_z(x) 
        tilde_h = self.linear_h(x) # Candidate state
        log_z = -F.softplus(-k) # Log (z) 
        log_one_minus_z = -F.softplus(k) # Log (1 - z)
        log_h_0 = self.log_g(h_0) # Hidden state at time 0
        log_tilde_h = self.log_g(tilde_h) # Log candidate state
        # Parallel scan (log z + log h_tilde since we are using log, and had z * h_tilde in the original implementation)
        h = parallel_scan_log(log_one_minus_z, torch.cat([log_h_0, log_z + log_tilde_h], dim=1))
        
        # Return h[1:t]
        return h


