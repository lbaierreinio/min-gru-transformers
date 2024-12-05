import os
import torch
import argparse
from transformers import AutoTokenizer
from dataclasses import dataclass
from utils.utility import create_file, append_line, get_new_row
from datasets.synthetic.utility import get_split
from train.utility import train
from models.TransformerSynthetic import TransformerSynthetic
from datasets.synthetic.generate_dataset import DatasetConfig

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

@dataclass
class TrainConfig:
    """
    Configuration for training.
    """
    learning_rate: float = 1e-4
    num_epochs: int = 2000
    early_stopping: bool = True
    num_classes: int = 8
    early_stopping_threshold: float = 0.95

def main():
    # (1) Retrieve arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        help='Training dataset path')
    
    parser.add_argument('--out_path', type=str,
                        help='Training dataset path')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_path = args.dataset_path
    out_path = args.out_path

    if not os.path.exists(out_path):
        create_file(args.out_path)

    # (2) Load Dataset
    dataset_config = DatasetConfig()

    tokenizer = AutoTokenizer.from_pretrained(dataset_config.tokenizer)

    dataset = torch.load(dataset_path)
    train_dataloader, val_dataloader = get_split(dataset)

    # (3) Load Training Parameters
    train_config = TrainConfig()
    loss_fn = torch.nn.CrossEntropyLoss()

    # (4) Define Model and Configuration
    vocab_size = tokenizer.vocab_size

    config = TransformerConfig()
    model = TransformerSynthetic(
        vocab_size=vocab_size,
        embedding_dim=config.embedding_dim,
        num_layers=config.num_layers,
        bidirectional=config.bidirectional,
        num_classes=train_config.num_classes,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    num_parameters = sum(p.numel() for p in model.parameters())

    # (5) Train Model
    best_training_loss, best_validation_loss, best_training_accuracy, best_validation_accuracy,validation_loss, validation_accuracy, steps, total_epochs, time_per_epoch, max_memory = train(
        model, train_dataloader, val_dataloader, train_config.num_epochs, loss_fn, optimizer, early_stopping_threshold=train_config.early_stopping_threshold)

    torch.save(model, f"{config.name}.pt")

    row = get_new_row()
    row['Model'] = config.name
    row['Layers'] = config.num_layers
    row['Parameters'] = num_parameters
    row['Dataset Path'] = dataset_path
    row['Training Steps'] = steps
    row['Number of Epochs'] = total_epochs
    row['Time Per Epoch'] = time_per_epoch
    row['Best Validation Accuracy'] = best_validation_accuracy
    row['Validation Accuracy'] = validation_accuracy
    row['Validation Loss'] = validation_loss
    row['Best Validation Loss'] = best_validation_loss
    row['Best Training Loss'] = best_training_loss
    row['Best Training Accuracy'] = best_training_accuracy
    row['Max Memory'] = max_memory

    append_line(out_path, row)

if __name__ == '__main__':
    main()
