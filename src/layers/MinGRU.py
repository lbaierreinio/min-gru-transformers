import torch
import torch.nn as nn
import torch.nn.functional as F

class MinGRU(nn.Module):
    def __init__(self, dim_in, dim_hidden):
        super().__init__()
        self.linear_z = nn.Linear(dim_in, dim_hidden) # Linear layer for producing z from x
        self.linear_h = nn.Linear(dim_in, dim_hidden) # Linear layer for producing candidate state h_tilde from x

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

    def g(self, x):
        """
        Appendix B.3: Were RNNs All We Needed?
        """
        return torch.where(x >= 0, x+0.5, torch.sigmoid(x))

    def log_g(self, x):
        """
        Appendix B.3: Were RNNs All We Needed?
        """
        return torch.where(x >= 0, torch.log(F.relu(x)+0.5), -F.softplus(-x))

    def forward(self, x, h_prev=None):
        """
        Compute the forward pass. Note that if h_prev is not none,
        then we assume the model is processing tokens sequentially.
        Otherwise, we enter parallel mode. In sequential mode,
        the sequence length should be 1. In parallel mode, the
        sequence length should be greater than 1. We return the
        output of the RNN and the hidden state if return_hidden is True.
        Args:
            x: torch.Tensor [batch_size, seq_len, dim_in]
            h_prev (optional): torch.Tensor [batch_size, seq_len, dim_hidden]

        Returns:
            h: torch.Tensor [batch_size, seq_len, dim_hidden]
        """
        k = self.linear_z(x)
        tilde_h = self.linear_h(x) # Candidate state

        if h_prev is not None: # Sequential mode
            assert x.shape[1] == 1
            z = torch.sigmoid(k)
            tilde_h = self.g(tilde_h)
            h = (1 - z) * h_prev + z * tilde_h # h[t]
        else: # Parallel Mode
            # NOTE: the implementation provided in the paper allows providing an explicit
            #       starting state h_0; we fix h_0 (implicitly) to be zero initialized
            log_z = -F.softplus(-k) # Log (z)
            log_one_minus_z = -F.softplus(k) # Log (1 - z)
            log_tilde_h = self.log_g(tilde_h) # Log candidate state
            h = self.parallel_scan_log(log_one_minus_z, log_z + log_tilde_h) # Hidden states

        return h