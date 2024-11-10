import torch
import torch.nn.functional as F
'''
TODO: Heinsen suggets that if coefficients are gating probabilities computed
from given logits, you should use F.logsigmoid(logits) insted of torch.log(F.sigmoid(logits))
'''
def parallel_scan_log(a, b):
    log_a = torch.log(a) 
    log_b = torch.log(b)
    # Cumsum returns a vector y where y_i = x1 + x2 + ... + x_i
    # F.pad w/ (1,0) returns the same vector, but with 0 appended at the beginning
    # We pad the first element so that we have y_0 = 0 instead of y_0 = x1
    a_star = F.pad(torch.cumsum(log_a, dim=-1), (1,0))

    # torch.logcumsumexp yields y_i = log(exp(x1) + exp(x2) + ... + exp(x_i))
    log_x0_plus_b_star = torch.logcumsumexp(log_b - a_star, dim=-1)

    log_x = a_star + log_x0_plus_b_star

    x = torch.exp(log_x)
    
    return x