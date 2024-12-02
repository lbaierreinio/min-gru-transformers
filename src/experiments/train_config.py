from dataclasses import dataclass


@dataclass
class TrainConfig:
    """
    Configuration for training.
    """
    learning_rate: float = 1e-4
    num_epochs: int = 500
    early_stopping: bool = True
    num_classes: int = 8
    early_stopping_threshold: float = 0.95
