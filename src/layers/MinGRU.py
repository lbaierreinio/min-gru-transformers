import torch
import torch.nn as nn
import torch.nn.functional as F

class MinGRU(nn.Module):
    def __init__(self, dim_h):
        super().__init__()
        self.linear_z = nn.Linear(dim_h, dim_h) # Linear layer for producing z from x
        self.linear_h = nn.Linear(dim_h, dim_h) # Linear layer for producing candidate state h_tilde from x

    def parallel_scan_log(self, log_a, log_b):
        """
        Given sequences log(a) and log(b) of length t, compute h[0:t-1],
        where h[0] = b[0], and h[i] = a[i]*h[i-1] + b[i] for i > 0.

        Args:
            log_a: torch.Tensor
            log_b: torch.Tensor

        Returns:
            h: torch.Tensor
        """
        # Take cumulative sum across seq_len dimension
        log_a_star = torch.cumsum(log_a, dim=1)
        # Obtain log(b) - a_star and take logcumsumexp across seq_len dimension
        log_x0_plus_b_star = torch.logcumsumexp(log_b - log_a_star, dim=1)

        log_x = log_a_star + log_x0_plus_b_star

        return log_x.exp()

    def log_g(self, x):
        """
        Appendix B.3: Were RNNs All We Needed?
        """
        return torch.where(x >= 0, torch.log(F.relu(x)+0.5), -F.softplus(-x))

    def forward(self, h_prev_layer):
        """
        Compute the forward pass. Note that if h_prev is not none,
        then we assume the model is processing tokens sequentially.
        Otherwise, we enter parallel mode. In sequential mode,
        the sequence length should be 1. In parallel mode, the
        sequence length should be greater than 1. We return the
        output of the RNN and the hidden state if return_hidden is True.
        Args:
            x: torch.Tensor
            h_prev: torch.Tensor
            return_hidden: bool

        Returns:
            out: torch.Tensor
            h: torch.Tensor
        """
        k = self.linear_z(h_prev_layer)
        tilde_h = self.linear_h(h_prev_layer) # Candidate state
        log_z = -F.softplus(-k) # Log (z)
        log_one_minus_z = -F.softplus(k) # Log (1 - z)
        log_tilde_h = self.log_g(tilde_h) # Log candidate state
        h = self.parallel_scan_log(log_one_minus_z, log_z + log_tilde_h) # Hidden states

        return h