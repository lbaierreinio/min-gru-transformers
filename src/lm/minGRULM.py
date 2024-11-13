from torch import nn
import torch.nn.functional as F
import torch.nn.modules.normalization as N
from src.minGRU.minGRU import MinGRU

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

class MinGRULM(nn.Module):
    """
    The MinGRULM class.
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
        conv_kernel_size=3,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_tokens = num_tokens

        self.embedding = nn.Embedding(num_tokens, input_dim)

        self.layers = nn.ModuleList([])

        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                CausalDepthWiseConv1D(input_dim, conv_kernel_size),
                N.RMSNorm(input_dim), # TODO: Verify behaviour with these arguments
                MinGRU(input_dim, hidden_dim),
                N.RMSNorm(input_dim), # TODO: Verify behaviour with these arguments
                FCNN(input_dim, hidden_dim)
            ]))

        # Final layer
        self.norm = N.RMSNorm(input_dim)
        self.out = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, prev_hiddens=None):
        """
        Forward pass of the model. In the sequential case, h_prev is
        all of the hidden states from the previous token across all layers.
        In the parallel case, h_prev should be none.
        Furthermore, in sequential case, seq_len should be 1.
        Args:
            x: torch.LongTensor, shape (batch_size, seq_len)    
            h_prev: torch.Tensor, shape (num_layers, hidden_dim)
        
        Returns:
            out: torch.Tensor, shape (batch_size, seq_len, hidden_dim)
            h_next: torch.Tensor, shape (batch_size, num_layers, hidden_dim)
        """
        x = self.embedding(x) # batch_size, sequence_length -> batch_size, sequence_length, hidden_dim

        h_next = [] # Stores the output hidden states from each layer of the deep RNN (which should be used in the next token)

        prev_hiddens_iter = iter(prev_hiddens if prev_hiddens is not None else []) # Iterate over num_layers dimension

        for conv, norm1, mingru, norm2, fcnn in self.layers: # Iterate over layers
            next_prev_hidden = next(prev_hiddens_iter) if prev_hiddens is not None else None
            # x = conv(x) + x # Convolution layer with skip connection
            min_gru_out, h_l_next = mingru(norm1(x), next_prev_hidden, return_hidden=(prev_hiddens is not None)) # MinGRU layer, using the previous hidden state from the appropriate layer.
            x = min_gru_out + x # Skip over MinGRU
            x = fcnn(norm2(x)) + x # Skip connection over RMSNorm & FCNN
            h_next.append(h_l_next) # Add hidden state from this layer
        
        # Compute embedding
        out = self.out(self.norm(x))

        return out, h_next # Return output token from output layer, and hidden states from each layer