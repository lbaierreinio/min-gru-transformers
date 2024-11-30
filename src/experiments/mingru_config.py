from dataclasses import dataclass


@dataclass
class MinGRUConfig:
    """
    Configuration for minGRU model.
    """
    name: str = 'mingru'
    embedding_dim: int = 256
    expansion_factor: float = 1.5
    num_layers: int = 4
    bidirectional: bool = True
