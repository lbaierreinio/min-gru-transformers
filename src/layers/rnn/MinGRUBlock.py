import torch.nn as nn
from layers.rnn.MinGRU import MinGRU
from layers.rnn.BiMinGRU import BiMinGRU


class MinGRUBlock(nn.Module):
    """A MinGRU Block (analogous to an attention block in functionality) """
    def __init__(self, hidden_dim, bidirectional=False):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.minGRU = BiMinGRU(hidden_dim, hidden_dim) if bidirectional else MinGRU(hidden_dim, hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, mask=None):
        x = x + self.minGRU(self.ln_1(x), mask=mask)
        x = x + self.mlp(self.ln_2(x))
        return x