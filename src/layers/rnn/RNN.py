import torch.nn as nn
from layers.rnn.MinGRU import MinGRU
from layers.rnn.BiMinGRU import BiMinGRU


class RNN(nn.Module):
    def __init__(self, *, embedding_dim, hidden_dim, num_layers=1, bidirectional=False):
        super().__init__()

        assert num_layers > 0, "Number of layers must be greater than 0"

        self.embedding_rnn = (BiMinGRU(embedding_dim, hidden_dim)
                              if bidirectional else MinGRU(embedding_dim, hidden_dim))

        self.layers = nn.ModuleList([
            (BiMinGRU(hidden_dim, hidden_dim) if bidirectional else MinGRU(hidden_dim, hidden_dim)) for _ in range(num_layers-1)
        ])

    def forward(self, x):
        x = self.embedding_rnn(x)
        for layer in self.layers:
            x = layer(x)

        return x
