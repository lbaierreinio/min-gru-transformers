import math
import torch.nn as nn
from layers.transformer.PositionalEncoding import PositionalEncoding
from layers.transformer.TransformerEncoderBlock import TransformerEncoderBlock

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, num_heads, num_layers, num_hiddens, ffn_num_hiddens, dropout, max_len, chunk_size=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.chunk_size = chunk_size
        self.pos_encoder = PositionalEncoding(num_hiddens, max_len)
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(num_heads, num_hiddens, ffn_num_hiddens, dropout) for _ in range(num_layers)
        ])
        self.num_hiddens = num_hiddens
        self.out = nn.GRU(num_hiddens, num_hiddens, num_layers=1, batch_first=True, bidirectional=False)

    def forward(self, x, mask=None, is_chunked=False):
        """
        Args:
            src: Tensor of shape [batch_size, seq_len]
            mask: ByteTensor of shape [batch_size, seq_len]
        """
         # Embedding + Positional Encoding
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
