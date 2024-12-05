import math
import torch.nn as nn
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

        # GRU for output
        self.out = nn.GRU(num_hiddens, num_hiddens, num_layers=1, batch_first=True)

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
            x_res = x_res[:, -1, :] # Extract last token's hidden state
            x_res = x_res.view(batch_size, num_chunks, num_hiddens) # (B, N, H)
        
        x, _ = self.out(x)

        return x[:, -1] # (B, H)
