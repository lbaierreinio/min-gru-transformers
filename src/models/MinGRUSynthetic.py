from torch import nn
from layers.rnn.MinGRUBlock import MinGRUBlock


class MinGRUSynthetic(nn.Module):
    def __init__(self, *, vocab_size, embedding_dim, num_layers=1, bidirectional=False, num_classes):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # MinGRU layers
        self.layers = nn.ModuleList(MinGRUBlock(embedding_dim) for _ in range(num_layers))

        # Classifier head
        self.linear = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.linear(x[:, -1]) # TODO: Update to use [CLS] token

        return x