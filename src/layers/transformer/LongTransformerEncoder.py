import math
import torch
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

    def forward(self, x, mask=None):

        x = self.embedding(x) * math.sqrt(self.num_hiddens)
        x = self.pos_encoder(x)

        num_chunks = int(x.shape[1] // self.chunk_size)
        batch_size, _, num_hiddens = x.shape

        x_chunks = x.view(batch_size, num_chunks,
                          self.chunk_size, num_hiddens)  # (Batch, Number of Chunks, Chunk Size, Hidden Dimension)
        x_chunks = x_chunks.transpose(0, 1)  # (N, B, C, H)
        x_chunks = x_chunks.reshape(-1, self.chunk_size, num_hiddens) # (N * B, C, H)

        if mask is not None:
            chunked_mask = mask.view(batch_size, num_chunks, self.chunk_size) # (B, N, C)
            chunked_mask = chunked_mask.transpose(0,1) # (N,B,C)
            chunked_mask = chunked_mask.reshape(batch_size * num_chunks, self.chunk_size) # (N * B, C)

        for layer in self.layers:
            x_chunks = layer(x_chunks, mask, chunked_mask if mask is not None else None)
        
        x_out = x_chunks.view(num_chunks, -1, self.chunk_size, num_hiddens) # (N, B, C, H)
        x_out = x_out.transpose(1, 0) # (B, N, C, H)
        x_out = x_out.reshape(batch_size, -1, num_hiddens) # (B, N * C, H)
        x_out = self.out(x_out) # (B, N * C, H)
        return x_out[:, -1] # (B, H)
