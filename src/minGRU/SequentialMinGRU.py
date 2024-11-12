import torch
import torch.nn as nn

class SequentialMinGRU(nn.Module):
    def __init__(self, dim_x, dim_h):
        super().__init__()
        self.linear_z = nn.Linear(dim_x, dim_h)
        self.linear_h = nn.Linear(dim_x, dim_h)
    
    def forward(self, x_t, h_prev):
        """
        Compute the forward pass.

        Args:
            x: torch.Tensor, shape (batch_size, seq_len, input_size)
            h_prev: torch.Tensor, shape (batch_size, 1, hidden_size)
        
        Returns:
            h_t: torch.Tensor, shape (batch_size, 1, hidden_size)
        """
        z_t = torch.sigmoid(self.linear_z(x_t)) # Output gate
        h_tilde = self.linear_h(x_t) # Candidate cell
        h_t = ((1 - z_t) * h_prev) + (z_t * h_tilde) # New hidden state
        return h_t