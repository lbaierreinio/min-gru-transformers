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

    def forward(self, x, mask=None, is_sequential=False):
        x = self.embedding(x)
        if is_sequential: # Sequential
            assert not self.bidirectional, "Bidirectional not supported in sequential mode"
            batch_size, seq_len, _ = x.shape
            h_prev = [torch.zeros(batch_size, self.embedding_dim) for _ in range(self.num_layers)]  # (L, B, E), hidden states from previous token
            for t in range(seq_len): # Iterate over tokens
                x_t = x[:, t, :] # (B, E) Current token
                for layer_idx, layer in enumerate(self.layers): # Iterate depthwise through the layers
                    mask_t = mask[:, t] if mask is not None else None # (B) Mask for current token
                    x_t, h_prev[layer_idx] = layer(x_t, mask=mask_t, h_prev=h_prev[layer_idx]) # Update hidden state
            
            return self.linear(x_t) # (B, C) Output logits from last token
        else: 
            for layer in self.layers:
                x = layer(x, mask=mask)
            return self.linear(x[:, -1])
