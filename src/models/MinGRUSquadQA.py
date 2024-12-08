from dataclasses import dataclass

import torch.nn as nn
from torch.nn import functional as F

from layers.rnn.MinGRUBlock import MinGRUBlock


@dataclass
class MinGRUSquadQAConfig:
    vocab_size: int = 30522 # Default BERT tokenizer vocab size
    n_layer: int = 12
    hidden_dim: int = 768
    classification_head_dim: int = 768
    dropout: int = 0.2
    bidirectional: bool = False


class MinGRUSquadQA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Produce contextualized embeddings for each sequence element
        self.encoder = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.hidden_dim),
            layers = nn.ModuleList(
                MinGRUBlock(
                    hidden_dim=config.hidden_dim,
                    bidirectional=config.bidirectional,
                    dropout=config.dropout
                ) for _ in range(config.n_layer)),
            ln_f = nn.LayerNorm(config.hidden_dim),
        ))
        # Project each embedding into 2D space representing [start_prob, end_prod]
        self.head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.classification_head_dim),
            nn.ReLU(),
            nn.Linear(config.classification_head_dim, 2)
        )

    def forward(self, x, targets=None, mask=None, is_sequential=False):
        B, T = x.shape

        x = self.encoder.wte(x) # (B, T, dim_hidden)
        # forward through layers of minGRU
        for layer in self.encoder.layers:
            x = layer(x, mask, is_sequential=is_sequential) # (B, T, dim_hidden)
        # final layernorm and classifier
        x = self.encoder.ln_f(x) # (B, T, dim_hidden)
        logits = self.head(x) # (B, T, 2)

        loss = self.loss(logits, targets, mask) if targets is not None else None
        return logits, loss

    def loss(self, logits, targets, mask=None):
        """Sum of cross entropy of start and end positions, weighed equally.

        Inputs:
        - logits [B, T, 2]: each [T, 2] item is a pair of logits for the start and end position for each token in the sequence
        - targets: [B, 2]: the correct start and end positions for each example
        - mask (optional): torch.Tensor [batch_size, seq_len]
        
        NOTE: mask is True for positions that should be masked out, False everywhere else
        """
        start_pos_logits = logits[:,:,0]
        start_pos_targets = targets[:,0]
        end_pos_logits = logits[:,:,1]
        end_pos_targets = targets[:,1]

        if mask is not None:
            # Set the logits for the masked positions to -inf. This will ensure that
            # they have no contribution in the cross entropy loss, since they will
            # be fed into a softmax first.
            start_pos_logits = start_pos_logits.masked_fill(mask, float("-inf"))
            end_pos_logits = end_pos_logits.masked_fill(mask, float("-inf"))
    
        return F.cross_entropy(start_pos_logits, start_pos_targets) + F.cross_entropy(end_pos_logits, end_pos_targets)
