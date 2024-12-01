from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """
    Configuration of the experiment.
    """
    sequence_length: int = 512
    num_examples: int = 100
    tokenizer: str = 'bert-base-uncased'
    alpha: int = 1
    beta: int = 4
    k_split: float = 0.05
    k_indicator: float = 0.1
