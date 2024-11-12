import torch
import torch.nn.functional as F

def parallel_scan_log(a, b):
    """
    Given batches of sequences a and b, where b[0] = h[0], 
    compute and return h[1:t] for each pair of sequences in the batch,
    where h[i] = a[i]*h[i-1] + b[i].

    Args:
        a: torch.Tensor, shape (batch_size, seq_len, hidden_size)
        b: torch.Tensor, shape (batch_size, seq_len+1, hidden_size)

    Returns:
        h: torch.Tensor, shape (batch_size, seq_len, hidden_size)
    """
    # Take log of a & b
    log_a = torch.log(a) 
    log_b = torch.log(b)
    # Take cumulative sum across seq_len dimension
    log_a_prime = torch.cumsum(log_a, dim=1)
    # Pad each sequence with zero vector at beginning with dimension hidden_size
    a_star = F.pad(log_a_prime, (0,0,1,0))
    # Obtain log(b) - a_star and take logcumsumexp across seq_len dimension
    log_x0_plus_b_star = torch.logcumsumexp(log_b - a_star, dim=1)

    log_x = a_star + log_x0_plus_b_star

    x = torch.exp(log_x)
    # Omit first element of each sequence, so that we return h1:t
    return x[:,1:]