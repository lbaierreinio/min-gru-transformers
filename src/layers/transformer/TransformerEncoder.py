import math
import torch.nn as nn
from layers.transformer.PositionalEncoding import PositionalEncoding
from layers.transformer.TransformerEncoderBlock import TransformerEncoderBlock

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, num_heads, num_layers, num_hiddens, ffn_num_hiddens, dropout, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoder = PositionalEncoding(num_hiddens, max_len)
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(num_heads, num_hiddens, ffn_num_hiddens, dropout) for _ in range(num_layers)
        ])
        self.num_hiddens = num_hiddens

    def forward(self, x, mask=None):
        """
        Args:
            src: Tensor of shape [batch_size, seq_len]
            mask: ByteTensor of shape [batch_size, seq_len]
        """
         # Embedding + Positional Encoding
        x = self.embedding(x) * math.sqrt(self.num_hiddens)
        x = self.pos_encoder(x)
        
        for layer in self.layers:
            x = layer(x, mask=mask)

        return x[:, -1]
