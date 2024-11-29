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

    self.rnn_out = nn.GRU(num_hiddens, num_hiddens, batch_first=True)

  def forward(self, x):
    x = self.embedding(x) * math.sqrt(self.num_hiddens)
    x = self.pos_encoder(x)

    num_chunks = int(x.shape[1] // self.chunk_size)
    batch_size, _, num_hiddens = x.shape

    x_chunks = x.view(batch_size, num_chunks, self.chunk_size, num_hiddens) # (B, N, C, H)
    x_chunks = x_chunks.transpose(0,1) # (N, B, C, H)
    x_chunks = x_chunks.reshape(-1, self.chunk_size, num_hiddens) # (N * B, C, H)

    x_out = x_chunks

    for layer in self.layers:
      x_out = layer(x_out)

    x_res = x_out[:, -1, :] # (N * B, H) Extract last token's hidden state
    x_res = x_res.view(num_chunks, batch_size, num_hiddens) # (N, B, H)
    x_res = x_res.transpose(0,1) # (B, N, H)

    x, _ = self.rnn_out(x) # (B, N, H)
    return x[:, -1]