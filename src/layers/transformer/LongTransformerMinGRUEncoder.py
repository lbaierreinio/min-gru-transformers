import math
import torch.nn as nn
from layers.transformer.TransformerMinGRUEncoderBlock import TransformerMinGRUEncoderBlock


class LongTransformerMinGRUEncoder(nn.Module):
    def __init__(self, vocab_size, num_heads, num_layers, num_hiddens=512, ffn_num_hiddens=2048, dropout=0.1, chunk_size=512, max_len=2048, bias=False):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.chunk_size = chunk_size

        # Embedding & Encoding layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)

        # Encoder Layers
        self.layers = nn.ModuleList([
            TransformerMinGRUEncoderBlock(num_heads, num_hiddens, ffn_num_hiddens, chunk_size, dropout, max_len, bias) for _ in range(num_layers)
        ])

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.num_hiddens)

        for layer in self.layers:
            x = layer(x)
        return x[:, -1]
