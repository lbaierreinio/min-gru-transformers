import os
import torch
import argparse
from transformers import AutoTokenizer
from dataclasses import dataclass
from datasets.synthetic.utility import get_split
from train.utility import train
from datasets.synthetic.generate_dataset import DatasetConfig
from experiments.mingru_config import MinGRUConfig
from experiments.transformer_config import TransformerConfig
from models.MinGRUSynthetic import MinGRUSynthetic
from models.TransformerSynthetic import TransformerSynthetic


@dataclass
class TrainConfig:
    """
    Configuration for training.
    """
    learning_rate: float = 1e-4
    num_epochs: int = 100
    early_stopping: bool = True
    num_classes: int = 8
    early_stopping_threshold: float = 0.95


def main():
    # (1) Retrieve arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        help='Training dataset path')
    parser.add_argument('--model', type=int,
                        help='Model to use: [0: MinGRU, 1: Transformer]')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_path = args.dataset_path
    model = args.model

    if dataset_path is None:
        raise ValueError("Paths must be specified")

    if model not in [0, 1]:
        raise ValueError("Model must be 0 or 1")

    if not os.path.exists(dataset_path):
        raise ValueError("Path must point to a valid file")

    # (2) Load Dataset
    dataset_config = DatasetConfig()

    tokenizer = AutoTokenizer.from_pretrained(
        dataset_config.tokenizer, model_max_length=dataset_config.sequence_length)

    dataset = torch.load(dataset_path)
    train_dataloader, val_dataloader = get_split(dataset)

    # (3) Load Training Parameters
    train_config = TrainConfig()
    loss_fn = torch.nn.CrossEntropyLoss()

    # (4) Define Model and Configuration
    vocab_size = tokenizer.vocab_size

    if model == 0:
        config = MinGRUConfig()
        model = MinGRUSynthetic(
            vocab_size=vocab_size,
            embedding_dim=config.embedding_dim,
            num_layers=config.num_layers,
            bidirectional=config.bidirectional,
            num_classes=train_config.num_classes,
        ).to(device)
    else:
        config = TransformerConfig()
        model = TransformerSynthetic(
            vocab_size=vocab_size,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            num_classes=train_config.num_classes,
            num_hiddens=config.num_hiddens,
            ffn_num_hiddens=config.ffn_num_hiddens,
            chunk_size=config.chunk_size,
            max_len=dataset_config.sequence_length,
        ).to(device)

    num_parameters = sum(p.numel() for p in model.parameters())

    # (5) Train Model
    validation_accuracy, total_validation_loss, steps, total_epochs, avg_time_per_step = train(
        model, train_dataloader, val_dataloader, train_config.num_epochs, loss_fn, train_config.learning_rate, early_stopping_threshold=train_config.early_stopping_threshold)

    torch.save(model, f"{config.name}.pt")


if __name__ == '__main__':
    main()
