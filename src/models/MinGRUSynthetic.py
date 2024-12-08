import torch
from torch import nn
from layers.rnn.MinGRUBlock import MinGRUBlock


class MinGRUSynthetic(nn.Module):
    def __init__(self, *, vocab_size, embedding_dim, num_layers=1, bidirectional=False, num_classes):
        super().__init__()
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # MinGRU layers
        self.layers = nn.ModuleList(MinGRUBlock(embedding_dim, bidirectional=bidirectional) for _ in range(num_layers))

        # Classifier head
        self.linear = nn.Linear(embedding_dim, num_classes)

    def forward(self, x, mask=None, is_sequential=False):
        x = self.embedding(x)
        if is_sequential: # Sequential
            assert not self.bidirectional, "Bidirectional not supported in sequential mode"
            batch_size, seq_len, _ = x.shape
            h_prev = [torch.zeros(self.num_layers, batch_size, self.embedding_dim)] # (L, B, E), hidden states from previous token
            for t in range(seq_len): # Iterate over tokens
                x_l = x[:, t, :] # (B, E) Current token
                for layer_idx, layer in enumerate(self.layers): # Iterate depthwise through the layers
                    h_prev_l = h_prev[layer_idx] # (B, E) previous hidden state for current layer
                    mask_t = mask[:, t] if mask is not None else None # (B) Mask for current token
                    h_prev[layer_idx] = layer(x_l, mask=mask_t, h_prev=h_prev_l) # Update hidden state
            return self.linear(h_prev[-1]) # (B, C) Output logits
        else: 
            for layer in self.layers:
                x = layer(x, mask=mask)
            return self.linear(x[:, -1])
