import os
import torch
import argparse
from transformers import AutoTokenizer
from dataclasses import dataclass
from datasets.synthetic.utility import get_split
from train.utility import train
from utils.utility import create_file, append_line, get_new_row
from datasets.synthetic.generate_dataset import DatasetConfig
from models.MinGRUSynthetic import MinGRUSynthetic
from models.TransformerSynthetic import TransformerSynthetic

@dataclass
class MinGRUConfig:
    """
    Configuration for minGRU model.
    """
    name: str = 'mingru'
    num_layers: int = 3
    embedding_dim: int = 384
    bidirectional: bool = True

@dataclass
class TransformerConfig:
    """
    Configuration for Transformer model.
    """
    name: str = 'transformer'
    num_hiddens = 512
    num_heads = 8
    num_layers = 4
    num_classes = 8
    ffn_num_hiddens = 2048
    dropout = 0.1
    max_len: int = 512

@dataclass
class TrainConfig:
    """
    Configuration for training.
    """
    learning_rate: float = 3e-4
    num_epochs: int = 500
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
    
    parser.add_argument('--model', type=int,
                        help='Model')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_path = args.dataset_path
    out_path = args.out_path
    model = args.model

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
            max_len=config.max_len,
            dropout=config.dropout
        ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)

    num_parameters = sum(p.numel() for p in model.parameters())

    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

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
    row['GPU'] = gpu_name

    append_line(out_path, row)

if __name__ == '__main__':
    main()
