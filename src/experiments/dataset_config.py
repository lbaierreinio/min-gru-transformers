from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """
    Configuration of an experiment.
    """
    sequence_length: int = 96
    num_examples: int = 500
    num_labels: int = 4
    num_subsequences: int = 2
    start: int = 0
    end: int = 96
    tokenizer: str = 'bert-base-uncased'
