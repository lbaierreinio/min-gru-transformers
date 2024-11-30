from dataclasses import dataclass


@dataclass
class MinGRUConfig:
    """
    Configuration for minGRU experiments.
    """
    sequence_length: int = 256
    num_examples: int = 2000
    num_labels: int = 4
    num_subsequences: int = 4
    start: int = 0
    end: int = 256
    tokenizer: str = 'bert-base-uncased'
    even: bool = False
