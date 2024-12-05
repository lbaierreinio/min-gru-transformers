import math
import torch.nn as nn
from layers.transformer.PositionalEncoding import PositionalEncoding

class LongTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, num_heads, num_layers, num_hiddens, ffn_num_hiddens, dropout, chunk_size, max_len):
        super().__init__()
        self.chunk_size = chunk_size
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoder = PositionalEncoding(num_hiddens, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_hiddens,
            nhead=num_heads,
            dim_feedforward=ffn_num_hiddens,
            dropout=dropout,
            batch_first = True,
            norm_first=True)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.rnn_out = nn.GRU(num_hiddens, num_hiddens, batch_first=True) # RNN Out Layer
        self.num_hiddens = num_hiddens

    def forward(self, x, mask=None):
        """
        Args:
            src: Tensor of shape [batch_size, seq_len]
            src_key_padding_mask: ByteTensor of shape [batch_size, seq_len]
        """
        # Embedding + Positional Encoding
        x = self.embedding(x) * math.sqrt(self.num_hiddens)
        x = self.pos_encoder(x)

        # Define shapes
        num_chunks = int(x.shape[1] // self.chunk_size)
        batch_size, seq_len, num_hiddens = x.shape

        # Chunk data
        x_chunks = x.view(batch_size, num_chunks, self.chunk_size, num_hiddens) # (B, N, C, E)
        x_chunks = x_chunks.reshape(-1, self.chunk_size, num_hiddens) # (N * B, C, E)

        # Chunk mask
        chunked_mask = mask.view(batch_size, num_chunks, self.chunk_size) # (B, N, C)
        chunked_mask = chunked_mask.reshape(-1, self.chunk_size) # (N * B, C)
        chunked_mask[chunked_mask.all(dim=1)] = False

        # Encoder layer
        x_out = self.transformer_encoder(x_chunks, src_key_padding_mask=chunked_mask) # (N * B, C, E)

        x_res = x.view(batch_size, num_chunks, self.chunk_size, num_hiddens) # (B, N, C, E)
        x_out = x_out.reshape(batch_size, -1, num_hiddens) # (B, N * C, E)

        # Compute aggregated mask    
        indices = [x+self.chunk_size-1 for x in range(0, seq_len, self.chunk_size)]
        aggr_mask = mask[:, indices] # (B, N)
        x_res = x_res.masked_fill(aggr_mask.unsqueeze(-1), 0) # (B, N, E)

        x_res, _ = self.rnn_out(x_res) # (B, N, E)
        return x_res[:, -1]
