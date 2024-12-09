import torch
from torch import nn
from layers.rnn.MinGRUBlock import MinGRUBlock


class MinGRUSynthetic(nn.Module):
    def __init__(self, *, vocab_size, embedding_dim, num_layers=1, bidirectional=False, num_classes):
        super().__init__()
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.bidirectional = bidirectional
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # MinGRU layers
        self.layers = nn.ModuleList(MinGRUBlock(embedding_dim, bidirectional=bidirectional) for _ in range(num_layers))

        # Classifier head
        self.linear = nn.Linear(embedding_dim, num_classes)

    def forward(self, x, mask=None, h_prev=None, return_out=True):
        x = self.embedding(x)
        if h_prev is not None: # Sequential
            assert not self.bidirectional, "Bidirectional not supported in sequential mode"
            for layer_idx, layer in enumerate(self.layers): # Iterate depthwise through the layers
                x, h_prev[layer_idx] = layer(x, mask=mask, h_prev=h_prev[layer_idx]) # Update hidden state
            
            if return_out:
                return self.linear(x), h_prev
            return h_prev
        else: 
            for layer in self.layers:
                x = layer(x, mask=mask)
            return self.linear(x[:, -1])
