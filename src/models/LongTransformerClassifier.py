import torch.nn as nn
from layers.transformer.LongTransformerEncoder import LongTransformerEncoder

class LongTransformerClassifier(nn.Module):
    def __init__(self, *, vocab_size, num_heads, num_layers, num_classes, num_hiddens=512, ffn_num_hiddens=2048, dropout=0.1, chunk_size=512, max_len=2048, bias=False):
        super().__init__()

        self.long_transformer_encoder = LongTransformerEncoder(vocab_size, num_heads, num_layers, num_hiddens, ffn_num_hiddens, dropout, chunk_size, max_len, bias)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(num_hiddens, num_classes)

    def forward(self, x):

        x = self.long_transformer_encoder(x)
        x = self.dropout(x)
        x = self.fc_out(x)
        return x