from dataclasses import dataclass, field
from typing import List, Dict, Tuple


@dataclass
class DatasetConfig:
    """
    Configuration of an experiment.
    """
    sequence_length: int = 128
    num_examples: int = 2000
    num_labels: int = 4
    num_subsequences: int = 4
    start: int = 0
    end: int = 128
    tokenizer: str = 'bert-base-uncased'
    even: bool = False
