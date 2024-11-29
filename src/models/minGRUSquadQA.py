from dataclasses import dataclass

import torch.nn as nn
from torch.nn import functional as F

from layers.MinGRU import MinGRU


@dataclass
class minGRUSquadQAConfig:
    vocab_size: int = 30522 # Default BERT tokenizer vocab size
    n_layer: int = 12
    hidden_dim: int = 768
    classification_head_dim: int = 768


class minGRULayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_dim)
        self.minGRU = MinGRU(config.hidden_dim, config.hidden_dim)
        self.ln_2 = nn.LayerNorm(config.hidden_dim)
        self.mlp = nn.Linear(config.hidden_dim, config.hidden_dim)

    def forward(self, x):
        x = x + self.minGRU(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class minGRUSquadQA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Produce contextualized embeddings for each sequence element
        self.encoder = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.hidden_dim),
            layers = nn.ModuleList(minGRULayer(config) for _ in range(config.n_layer)),
            ln_f = nn.LayerNorm(config.hidden_dim),
        ))
        # Project each embedding into 2D space representing [start_prob, end_prod]
        self.head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.classification_head_dim),
            nn.ReLU(),
            nn.Linear(config.classification_head_dim, 2)
        )

    def forward(self, x, targets=None):
        B, T = x.shape

        x = self.encoder.wte(x) # (B, T, dim_hidden)
        # forward the blocks of the transformer
        for layer in self.layers.h:
            x = layer(x) # (B, T, dim_hidden)
        # final layernorm and classifier
        x = self.encoder.ln_f(x) # (B, T, dim_hidden)
        logits = self.head(x) # (B, T, 2)

        loss = self.loss(logits, targets) if targets is not None else None
        return logits, loss

    def loss(self, logits, targets):
        raise Exception, "TODO"