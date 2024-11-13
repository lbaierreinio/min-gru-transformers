import torch
import torch.nn.functional as F

def parallel_scan_log(log_a, log_b):
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