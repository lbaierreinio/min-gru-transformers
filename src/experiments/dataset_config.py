from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """
    Configuration of an experiment.
    """
    sequence_length: int = 192
    num_examples: int = 2000
    num_labels: int = 4
    num_subsequences: int = 4
    start: int = 0
    end: int = 192
    tokenizer: str = 'bert-base-uncased'
    even: bool = False
