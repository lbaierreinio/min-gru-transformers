import torch.nn as nn

class TransformerEncoderBlock(nn.Module):
    """
    A single Layer of the Transformer Encoder.
    """
    def __init__(self, num_heads, num_hiddens, ffn_num_hiddens, dropout):
        """
        Args:
            num_heads: int
                The number of attention heads.
            num_hiddens: int
                The number of hidden units.
            ffn_num_hiddens: int
                The number of hidden units in the feedforward network.
            dropout: float
                The dropout probability.
        """
        super().__init__()
        # Layer Normalization
        self.layernorm1 = nn.LayerNorm(num_hiddens)
        self.layernorm2 = nn.LayerNorm(num_hiddens)

        # Attention
        self.attention = nn.MultiheadAttention(
            num_hiddens, num_heads, dropout=dropout, batch_first=True)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(num_hiddens, ffn_num_hiddens),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(ffn_num_hiddens, num_hiddens)
        )

    def forward(self, x, mask=None):
        norm_x = self.layernorm1(x)
        attn_output, _ = self.attention(norm_x, norm_x, norm_x, key_padding_mask=mask)
        x = x + self.dropout1(attn_output)

        norm_x = self.layernorm2(x)
        ffn_output = self.ffn(norm_x)
        x = x + self.dropout2(ffn_output)

        return x
