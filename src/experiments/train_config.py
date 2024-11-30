from dataclasses import dataclass


@dataclass
class TrainConfig:
    """
    Configuration for training.
    """
    learning_rate: float = 1e-4
    num_epochs: int = 200
    early_stopping: bool = True
