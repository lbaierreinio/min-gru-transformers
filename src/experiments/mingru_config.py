from dataclasses import dataclass


@dataclass
class MinGRUConfig:
    """
    Configuration for minGRU model.
    """
    name: str = 'mingru'
    embedding_dim: int = 768
    expansion_factor: float = 2.5
    num_layers: int = 4
    bidirectional: bool = True
