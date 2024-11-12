from torch import nn
import torch.nn.functional as F
import torch.nn.modules.normalization as N

from src.minGRU.ParallelMinGRU import ParallelMinGRU


class FCNN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, x):
        return self.net(x)

class CausalDepthWiseConv1D(nn.Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding='same', groups=dim),
            nn.Conv1d(dim, dim, kernel_size=1)
        )
    
    # TODO: Figure out why we need to transpose
    def forward(self, x):
        x = x.transpose(1, 2) # b n d -> b d n
        x = self.net(x)
        return x.transpose(1, 2) # b d n -> b n d

class minGRULM(nn.Module):
    """
    The minGRULM class.
    args:
        num_tokens: int, the number of tokens in the vocabulary
        input_dim: int, the dimension of each token in the input sequence
        hidden_dim: int, the dimension of the hidden state
        num_layers: int, the depth of the model
        conv_kernel_size: int, the kernel size of the convolutional layer
    """
    def __init__(self,
        *,
        num_tokens,
        input_dim,
        hidden_dim,
        num_layers,
        conv_kernel_size=3 
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
                CausalDepthWiseConv1D(input_dim, conv_kernel_size),
                N.RMSNorm(input_dim), # TODO: Verify behaviour with these arguments
                ParallelMinGRU(input_dim, hidden_dim),
                N.RMSNorm(input_dim), # TODO: Verify behaviour with these arguments
                FCNN(input_dim, hidden_dim)
            ]))

        # Final layer
        self.norm = N.RMSNorm(input_dim)
        self.out = nn.Linear(input_dim, num_tokens)

    def forward(self, x, h_prev):
        """
        Forward pass of the model.
        args:
            x: torch.Tensor, shape (batch_size, seq_len, input_size)
            h_prev: torch.Tensor, shape (batch_size, 1, hidden_size). Note that in parallel mode, h_prev is always h[0].
        """
        x = self.embedding(x)

        # Keep passing prev_hidden to the next layer
        for conv, norm1, mingru, norm2, fcnn in self.layers: # Iterate over layers
            x = conv(x) + x # Convolution layer with skip connection
            x = mingru(norm1(x), h_prev) + x # Skip connection over RMSNorm & MinGRU
            x = fcnn(norm2(x)) + x # Skip connection over RMSNorm & FCNN
        
        # Compute logits
        logits = self.out(self.norm(x))

        return logits
