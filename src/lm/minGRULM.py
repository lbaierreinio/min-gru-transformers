import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.modules.normalization as N

from minGRU.ParallelMinGRU import ParallelMinGRU

def FCLayer(dim, hidden_dim):
    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.GeLU(), # TODO: Can consider switching activation function
        nn.Linear(hidden_dim, dim)
    )

class minGRULM(nn.Module):
    """
    The minGRULM class.
    args:
        num_tokens: int, the number of tokens in the vocabulary
        dim: int, the dimension of the hidden state
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
        self.embedding = nn.Embedding(num_tokens, input_dim)

        self.layers = nn.ModuleList([])

        for _ in range(num_layers):
            self.layers.append([
                # TODO: Add CausalDepthWiseConv1D
                N.RMSNorm(input_dim), # TODO: Verify
                ParallelMinGRU(input_dim, hidden_dim),
                N.RMSNorm(input_dim), # TODO: Verify
                FCLayer(input_dim, hidden_dim)
            ])

    def forward(self, x):
        """
        x becomes each input except for the last token (predicting tokens x[1:t] using x[0:t-1])
        labels becomes each input except for the first token 
        TODO: This is the code from the paper, they are doing
        next word prediction. In our case, we are just interested
        in getting an embedding.
        """
        # We are going forward through a sequence of tokens
        # But we are also going deep through the layers
        # Recall that mingru outputs h[1:t]. So what do we do with h[0]?

        # Keep passing prev_hidden to the next layer
        for layer in self.layers:
            # TODO: Figure out how hidden states flow through layers
            # TODO: Figure out how to handle h[0] & h[1:t]
            # TODO: Add residual connections
            # TODO: Define loss function
            pass

        # TODO: Return loss here.
        loss = None
        
        return loss
