from torch import nn
import torch.nn.functional as F
import torch.nn.modules.normalization as N

from minGRU.minGRU import MinGRU

class minGRUClassification(nn.Module):
  '''
  A simple 1-layer RNN using minGRU with a classifier head.
  As an example, for the SMS classification task, we used an 
  embedding dimension of 64, expansion factor of 2, and 2 classes.
  '''
  def __init__(self, vocab_size, embedding_dim, expansion_factor, num_classes):
    super().__init__()
    self.dim_inner = int(embedding_dim * expansion_factor)
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    self.rnn = MinGRU(embedding_dim, self.dim_inner)
    self.linear = nn.Linear(embedding_dim, num_classes) # Classifier head

  def forward(self, x):
    x = self.embedding(x)
    x = self.rnn(x)
    x = self.linear(x[:, -1])

    return x