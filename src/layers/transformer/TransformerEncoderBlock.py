import torch.nn as nn

class TransformerEncoderBlock(nn.Module):
  def __init__(self, num_heads, num_hiddens, ffn_num_hiddens, dropout, bias=False):
    super().__init__()
    # Layer Normalization
    self.layernorm1 = nn.LayerNorm(num_hiddens)
    self.layernorm2 = nn.LayerNorm(num_hiddens)

    # Attention
    self.attention = nn.MultiheadAttention(num_hiddens, num_heads, dropout=dropout, bias=bias)

    # Dropout
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)

    # Position wise FFN
    self.ffn1 = nn.Linear(num_hiddens, ffn_num_hiddens, bias=bias)
    self.relu = nn.ReLU()
    self.ffn2 = nn.Linear(ffn_num_hiddens, num_hiddens, bias=bias) 

  def forward(self, x):
    skip1 = x
    x, _ = self.attention(x,x,x) # Self Attention (Sublayer One)
    skip2 = self.layernorm1(self.dropout1(x) + skip1)
    x = self.ffn2(self.relu(self.ffn1(skip2))) # Position-Wise FNN (Sublayer 2)
    return self.layernorm2(self.dropout2(x) + skip2)