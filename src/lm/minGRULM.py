from torch import nn
import torch.nn.functional as F
import torch.nn.modules.normalization as N

from src.minGRU.ParallelMinGRU import ParallelMinGRU

def FCLayer(dim, hidden_dim):
    """
    Fully connected layer.
    args:
        dim: int, the dimension of the input.
        hidden_dim: int, the dimension of the hidden layer.
    """
    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.GELU(), # TODO: Verify use of activation function and/or consider alternatives.
        nn.Linear(hidden_dim, dim)
    )

class minGRULM(nn.Module):
    """
    The minGRULM class.
    args:
        num_tokens: int, the number of tokens in the vocabulary
        input_dim: int, the dimension of each token in the input sequence
        hidden_dim: int, the dimension of the hidden state
        num_layers: int, the depth of the model
    """
    def __init__(self,
        *,
        num_tokens,
        input_dim,
        hidden_dim,
        num_layers
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_tokens = num_tokens
        """
        nn.Embedding:
        A simple lookup table that stores embeddings of a fixed dictionary and size.
        """
        self.embedding = nn.Embedding(num_tokens, input_dim)

        self.layers = nn.ModuleList([])

        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                # TODO: Add CausalDepthWiseConv1D
                N.RMSNorm(input_dim), # TODO: Verify behaviour with these arguments
                ParallelMinGRU(input_dim, hidden_dim),
                N.RMSNorm(input_dim), # TODO: Verify behaviour with these arguments
                FCLayer(input_dim, hidden_dim)
            ]))

        # Final layer
        self.norm = N.RMSNorm(input_dim)
        self.out = nn.Linear(input_dim, num_tokens)

    def forward(self, x, h_prev):
        """
        Forward pass of the model.
        args:
            x: torch.Tensor, shape (batch_size, seq_len, input_size)
            h_prev: torch.Tensor, shape (batch_size, 1, hidden_size)
        """
        # x = self.embedding(x)

        # Keep passing prev_hidden to the next layer
        for norm1, mingru, norm2, fcnn in self.layers:
            # TODO: Add convolutional layer
            x = mingru(norm1(x), h_prev) + x # Skip connection over RMSNorm & MinGRU
            x = fcnn(norm2(x)) + x # Skip connection over RMSNorm & FCNN
        
        # Compute logits
        logits = self.out(self.norm(x))

        return logits
