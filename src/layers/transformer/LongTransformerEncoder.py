import math
import torch.nn as nn
from layers.rnn.BiMinGRU import BiMinGRU
from layers.transformer.PositionalEncoding import PositionalEncoding
from layers.transformer.TransformerEncoderBlock import TransformerEncoderBlock


class LongTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, num_heads, num_layers, num_hiddens=512, ffn_num_hiddens=2048, dropout=0.1, chunk_size=512, max_len=2048, bias=False):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.chunk_size = chunk_size

        # Embedding & Encoding
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoder = PositionalEncoding(num_hiddens, max_len=max_len)

        # Encoder Layers
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(num_heads, num_hiddens, ffn_num_hiddens, dropout, bias) for _ in range(num_layers)
        ])

        # MinGRU for output
        self.out = BiMinGRU(num_hiddens, num_hiddens)

    def forward(self, x, mask=None, is_chunked=False):
        x = self.embedding(x) * math.sqrt(self.num_hiddens)
        x = self.pos_encoder(x)

        if is_chunked:
            batch_size, max_seq_len, num_hiddens = x.shape
            num_chunks = int(max_seq_len // self.chunk_size)
            x = x.view(batch_size, num_chunks, self.chunk_size, num_hiddens)  # (Batch, Number of Chunks, Chunk Size, Hidden Dimension)
            x = x.reshape(-1, self.chunk_size, num_hiddens) # (N * B, C, H)
            
            chunked_mask = None
            if mask is not None:
                chunked_mask = mask.view(batch_size, num_chunks, self.chunk_size)
                chunked_mask = chunked_mask.reshape(batch_size * num_chunks, self.chunk_size)
                chunked_mask[chunked_mask.all(dim=1)] = False # Attend to rows that are exclusively padding tokens (as they will be masked out later)

        for layer in self.layers:
            x = layer(x, mask=chunked_mask if is_chunked else mask) # (N * B, C, H)
        
        if is_chunked:
            x = x.view(num_chunks, -1, self.chunk_size, num_hiddens) # (N, B, C, H)
            x = x.reshape(batch_size, -1, num_hiddens) # (B, N * C, H)
            x = self.out(x, mask) # (B, N * C, H)

        return x[:, -1] # (B, H)
