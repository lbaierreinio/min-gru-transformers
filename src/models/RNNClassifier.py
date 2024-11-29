from torch import nn
from layers.rnn.RNN import RNN

class RNNClassifier(nn.Module):
  def __init__(self, *, vocab_size, embedding_dim, expansion_factor, num_layers=1, bidirectional=False, num_logits):
    super().__init__()
    self.hidden_dim = int(embedding_dim * expansion_factor)

    # Embedding layer
    self.embedding = nn.Embedding(vocab_size, embedding_dim)

    # RRN layer (potentially deep and/or bidirectional)
    self.rnn = RNN(embedding_dim=embedding_dim, hidden_dim=self.hidden_dim, num_layers=num_layers, bidirectional=bidirectional) 

    # Classifier head
    self.linear = nn.Linear(self.hidden_dim, num_logits) 

  def forward(self, x):
    x = self.embedding(x)
    x = self.rnn(x)
    x = self.linear(x[:, -1])

    return x