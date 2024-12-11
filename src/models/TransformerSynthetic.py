import torch.nn as nn
from layers.transformer.TransformerEncoder import TransformerEncoder


class TransformerSynthetic(nn.Module):
    """
    Transformer model for the synthetic dataset that uses the Long Context
    Transformer Encoder with a Linear Classification Head.
    """
    def __init__(self, *, vocab_size, num_heads, num_layers, num_classes, num_hiddens=128, ffn_num_hiddens=512, dropout=0.1, max_len=2048, chunk_size=None):
        """
        Args:
            vocab_size: int
                The size of the vocabulary.
            num_heads: int
                The number of attention heads.
            num_layers: int
                The number of transformer layers.
            num_classes: int
                The number of classes.
            num_hiddens: int
                The number of hidden units.
            ffn_num_hiddens: int
                The number of hidden units in the feedforward network.
            dropout: float
                The dropout probability.
            max_len: int
                The maximum sequence length.
            chunk_size: int
                The chunk size for chunked attention. Default is None.
        """
        super().__init__()

        self.transformer_encoder = TransformerEncoder(
            vocab_size, num_heads, num_layers, num_hiddens, ffn_num_hiddens, dropout, max_len, chunk_size)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(num_hiddens, num_classes)

    def forward(self, x, mask=None):
        x = self.transformer_encoder(x, mask)
        x = self.dropout(x)
        x = self.fc_out(x)
        return x
