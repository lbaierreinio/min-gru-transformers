from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """
    Configuration for Transformer model.
    """
    name: str = 'transformer'
    num_heads: int = 4
    num_layers: int = 4
    num_hiddens: int = 128
    ffn_num_hiddens: int = 512
    chunk_size: int = 32
