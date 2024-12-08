import torch
import torch.nn as nn
import torch.nn.functional as F


class MinGRU(nn.Module):
    def __init__(self, dim_in, dim_hidden):
        super().__init__()
        # Linear layer for producing z from x
        self.linear_z = nn.Linear(dim_in, dim_hidden)
        # Linear layer for producing candidate state h_tilde from x
        self.linear_h = nn.Linear(dim_in, dim_hidden)
        self.dim_hidden = dim_hidden
        self.dim_in = dim_in

    def parallel_scan_log(self, log_a, log_b, mask=None):
        """
        Given sequences log(a) and log(b) of length t, compute h[0:t-1],
        where h[0] = b[0], and h[i] = a[i]*h[i-1] + b[i] for i > 0.

        Args:
            log_a: torch.Tensor
            log_b: torch.Tensor

        Returns:
            h: torch.Tensor
        """
        if mask is not None:
            log_a = log_a.masked_fill(mask, 0)
            log_b = log_b.masked_fill(mask, 0)

        # Take cumulative sum across seq_len dimension
        log_a_star = torch.cumsum(log_a, dim=1)

        log_a_b_star = log_b - log_a_star

        if mask is not None:
            log_a_b_star = log_a_b_star.masked_fill(mask, float('-inf'))

        log_x0_plus_b_star = torch.logcumsumexp(log_a_b_star, dim=1)
        
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

    def forward(self, x, *, mask=None, is_sequential = False):
        """
        Compute the forward pass. Note that if h_prev is not none,
        then we assume the model is processing tokens sequentially.
        Otherwise, we enter parallel mode. In sequential mode,
        the sequence length should be 1. In parallel mode, the
        sequence length should be greater than 1. We return the
        output of the RNN and the hidden state if return_hidden is True.

        If mask is provided, mask out the hidden states corresponding to the True positions
        in the mask (this is used to mask out padding tokens in the sequence).

        Args:
            x: torch.Tensor [batch_size, seq_len, dim_in]
            mask (optional): torch.Tensor [batch_size, seq_len]
            is_sequential (optional): bool
        Returns:
            h: torch.Tensor [batch_size, seq_len, dim_hidden]
        """
        k = self.linear_z(x)
        tilde_h = self.linear_h(x)  # Candidate state

        if is_sequential:  # Sequential mode
            batch_size, seq_len, _ = x.shape
            h_prev = torch.zeros(batch_size, self.dim_hidden)
            h = torch.zeros(batch_size, 0, self.dim_hidden)

            for t in range(seq_len): # Iterate over sequence length
                k_t = k[:, t, :]
                tilde_h_t = tilde_h[:, t, :]
                z_t = torch.sigmoid(k_t)
                tilde_h_t_p = self.g(tilde_h_t)
                h_prev = ((1 - z_t) * h_prev) + (z_t * tilde_h_t_p)
                if mask is not None:
                    mask_t = mask[:, t]
                    h_prev = h_prev.masked_fill(mask_t.unsqueeze(-1), 0)
                h = torch.cat((h, h_prev.unsqueeze(1)), dim=1)
        else:  # Parallel Mode
            # NOTE: the implementation provided in the paper allows providing an explicit
            #       starting state h_0; we fix h_0 (implicitly) to be zero initialized
            log_z = -F.softplus(-k)  # Log (z)
            log_one_minus_z = -F.softplus(k)  # Log (1 - z)
            log_tilde_h = self.log_g(tilde_h)  # Log candidate state

            h = self.parallel_scan_log(
                log_one_minus_z, log_z + log_tilde_h, mask.unsqueeze(-1) if mask is not None else None)  # Hidden states

        if mask is not None:
            mask = mask.unsqueeze(-1)
            h = h.masked_fill(mask, 0)
        return h
