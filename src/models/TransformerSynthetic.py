import torch.nn as nn
from layers.transformer.TransformerEncoder import TransformerEncoder


class TransformerSynthetic(nn.Module):
    def __init__(self, *, vocab_size, num_heads, num_layers, num_classes, num_hiddens=128, ffn_num_hiddens=512, dropout=0.1, max_len=2048, chunk_size=None):
        super().__init__()

        self.transformer_encoder = TransformerEncoder(
            vocab_size, num_heads, num_layers, num_hiddens, ffn_num_hiddens, dropout, max_len, chunk_size)
        
        self.classification_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_hiddens, num_classes),
        )
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(num_hiddens, num_classes)

    def forward(self, x, mask=None):
        x = self.transformer_encoder(x, mask)
        x = self.classification_head(x)
        return x
