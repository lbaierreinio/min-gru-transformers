import torch
import torch.nn.functional as F

def parallel_scan(a, b):
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
    # Take cumulative sum across seq_len dimension
    a_prime = torch.cumsum(a, dim=1)
    # Pad each sequence with zero vector at beginning with dimension hidden_size
    a_star = F.pad(a_prime, (0,0,1,0))
    # Obtain log(b) - a_star and take logcumsumexp across seq_len dimension
    x0_plus_b_star = torch.logcumsumexp(b - a_star, dim=1)

    x = a_star + x0_plus_b_star

    return x[:, 1:]