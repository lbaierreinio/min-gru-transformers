from dataclasses import dataclass


@dataclass
class MinGRUConfig:
    """
    Configuration for minGRU model.
    """
    name: str = 'mingru'
    embedding_dim: int = 256
    expansion_factor: float = 2.5
    num_layers: int = 3
    bidirectional: bool = True