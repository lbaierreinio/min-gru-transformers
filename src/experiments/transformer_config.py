from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """
    Configuration for Transformer model.
    """
    name: str = 'transformer'
    num_heads: int = 2
    num_layers: int = 2
    num_hiddens: int = 32
    ffn_num_hiddens: int = 128
    chunk_size: int = 128
