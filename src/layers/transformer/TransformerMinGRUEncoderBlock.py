import torch.nn as nn
from layers.rnn.BiMinGRU import BiMinGRU
from layers.transformer.PositionalEncoding import PositionalEncoding
from layers.transformer.TransformerEncoderBlock import TransformerEncoderBlock

class TransformerMinGRUEncoderBlock(nn.Module):
  def __init__(self, num_heads, num_hiddens, ffn_num_hiddens, chunk_size, dropout, max_len, bias=False):
    super().__init__()
    self.chunk_size = chunk_size

    self.transformer_encoder = TransformerEncoderBlock(num_heads, num_hiddens, ffn_num_hiddens, dropout, bias)
    self.bi_min_gru = BiMinGRU(num_hiddens, num_hiddens)
    self.ln = nn.LayerNorm(num_hiddens)
    self.pos_encoder = PositionalEncoding(num_hiddens, max_len=max_len)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    x = self.pos_encoder(x)
    batch_size, seq_len, num_hiddens = x.shape

    num_chunks = int(seq_len // self.chunk_size) # Compute number of chunks

    x_chunks = x.view(batch_size, num_chunks, self.chunk_size, num_hiddens) # (B, N, C, H)
    x_chunks = x_chunks.transpose(0,1) # (N, B, C, H)
    x_chunks = x_chunks.reshape(-1, self.chunk_size, num_hiddens) # (N * B, C, H)

    x_out = self.transformer_encoder(x_chunks)

    x_out = x_out.view(num_chunks, batch_size, self.chunk_size, num_hiddens) # (N, B, C, H)
    x_out = x_out.transpose(1, 0) # (B, N, C, H)
    x_out = x_out.reshape(batch_size, -1, num_hiddens) # (B, S, H)
    x = self.bi_min_gru(x_out)
    return self.ln(self.dropout(x) + x_out) # Skip connection & layer_norm around BiMinGRU
    