from torch import nn
import torch.nn.functional as F
import torch.nn.modules.normalization as N
from minGRU.minGRU import MinGRU

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
        x = x.transpose(-2, -1) # b n d -> b d n
        x = self.net(x)
        return x.transpose(-2, -1) # b d n -> b n d

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
        Forward pass of the model. In the sequential case, prev_hiddens should 
        be a list of hidden states from the previous token, for all layers. If 
        this is the first token in the sequence, prev_hiddens can be a list of
        zero tensors. Furthermore, in the sequential case, x should be only one
        sequence in the tensor. In the parallel case, prev_hiddens should be None,
        and x should be a tensor of sequences. The output is the embedding of all
        the tokens in the sequence, and the hidden states from each layer. If
        we are in parallel mode, we have no use for the hidden states.

        Args:
            x: torch.LongTensor
            h_prev: torch.Tensor
        
        Returns:
            out: torch.Tensor
            h_next: torch.Tensor
        """
        is_sequential = prev_hiddens is not None
        
        x = self.embedding(x) # batch_size, sequence_length -> batch_size, sequence_length, hidden_dim

        h_next = [] # Stores the output hidden states from each layer of the deep RNN (which should be used in the next token)

        prev_hiddens_iter = iter(prev_hiddens if is_sequential else []) # Iterate over num_layers dimension

        for conv, norm1, mingru, norm2, fcnn in self.layers: # Iterate over layers
            next_prev_hidden = next(prev_hiddens_iter) if is_sequential else None
            x = conv(x) + x # Convolution layer with skip connection
            # MinGRU layer, using the previous hidden state from the appropriate layer.
            min_gru_out, h_l_next = mingru(norm1(x), next_prev_hidden, return_hidden=is_sequential)
            x = min_gru_out + x # Skip over MinGRU
            x = fcnn(norm2(x)) + x # Skip connection over RMSNorm & FCNN
            h_next.append(h_l_next) # Add hidden state from this layer
        
        # Compute embedding
        out = self.out(self.norm(x))

        return out, h_next # Return output token from output layer, and hidden states from each layer