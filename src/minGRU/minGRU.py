import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.utility import parallel_scan_log

class MinGRU(nn.Module):
    def __init__(self, dim_x, dim_h): # Note: MinGRU paper suggests embedding dimension of 128
        super().__init__()
        self.linear_z = nn.Linear(dim_x, dim_h) # Linear layer for producing z from x
        self.linear_h = nn.Linear(dim_x, dim_h) # Linear layer for producing candidate state h_tilde from x
        self.linear_o = nn.Linear(dim_h, dim_x) # Linear layer for producing output from hidden state
    
    def log_g(self, x):
        """
        Appendix B.3: Were RNNs All We Needed?
        """
        return torch.where(x >= 0, torch.log(F.relu(x)+0.5), -F.softplus(-x))

    def forward(self, x, h_prev=None, *, return_hidden=False):
        """
        Compute the forward pass. Note that if seq_len of x is 1,
        then we assume the model is processing tokens sequentially.
        Otherwise, if we pass multiple tokens per batch, we enter
        parallel mode. In sequential mode, h_prev is the hidden state
        from the previous token (or initial hidden state), while in 
        parallel mode, h_prev is necessarily the initial hidden state.
        Args:
            x: torch.Tensor, shape (batch_size, seq_len, input_size)
            h_prev: torch.Tensor, shape (1, hidden_size)
        
        Returns:
            out: torch.Tensor, shape (batch_size, seq_len, input_size)
            h: torch.Tensor, shape (batch_size, (1/seq_len), input_size)
        """
        k = self.linear_z(x) 
        tilde_h = self.linear_h(x) # Candidate state

        if h_prev is not None: # Indicates sequential mode
            z = torch.sigmoid(k)
            h = (1 - z) * h_prev + z * tilde_h # h[t]
        else: # Parallel Mode (TODO: Determine if we would ever be interested in previous hidden state here)
            log_z = -F.softplus(-k) # Log (z) 
            log_one_minus_z = -F.softplus(k) # Log (1 - z)
            log_tilde_h = self.log_g(tilde_h) # Log candidate state
            h = parallel_scan_log(log_one_minus_z, log_z + log_tilde_h) # Hidden states
                 
        # Transform hidden state to output
        out = self.linear_o(h)

        if return_hidden:
            return out, h
        return out