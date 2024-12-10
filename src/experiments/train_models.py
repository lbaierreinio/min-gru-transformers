import os
import torch
import argparse
from transformers import AutoTokenizer
from models.MinGRUSynthetic import MinGRUSynthetic
from models.TransformerSynthetic import TransformerSynthetic
from utils.utility import create_file, append_line, get_new_row
from datasets.synthetic.utility import get_split
from dataclasses import dataclass
from train.utility import train
import matplotlib.pyplot as plt

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
    num_hiddens = 256
    num_heads = 8
    num_layers = 6
    ffn_num_hiddens = 1024
    dropout = 0.1
    chunk_size: int = 512
    max_len: int = 2048

@dataclass
class TrainConfig:
    """
    Configuration for training.
    """
    learning_rate: float = 3e-4
    num_epochs: int = 100
    early_stopping: bool = True
    num_classes: int = 8
    early_stopping_threshold: float = 0.95
    tokenizer: str = 'bert-base-uncased'
    accumulate_every_i: int = 4

"""
Script to generate and save a synthetic dataset given the current state
of the dataset configuration file.
"""
def main():
    transformer_config = TransformerConfig()
    mingru_config = MinGRUConfig()
    train_config = TrainConfig()

    tokenizer = AutoTokenizer.from_pretrained(train_config.tokenizer)
    tokenizer.padding_side = "left"

    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', type=str)
    parser.add_argument('--transformer_dataset_path', type=str)
    parser.add_argument('--mingru_dataset_path', type=str)

    args = parser.parse_args()

    out_path = args.out_path
    transformer_dataset_path= args.transformer_dataset_path
    mingru_dataset_path = args.mingru_dataset_path

    if not os.path.exists(out_path):
        create_file(args.out_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    print(f"Using device: {gpu_name}")

    transformer_dataset = torch.load(transformer_dataset_path)
    transformer_train_dataloader, transformer_val_dataloader = get_split(transformer_dataset, batch_size=8)
    # Train Transformer
    transformer = TransformerSynthetic(
        vocab_size=tokenizer.vocab_size,
        num_heads=transformer_config.num_heads,
        num_layers=transformer_config.num_layers,
        num_classes=train_config.num_classes,
        num_hiddens=transformer_config.num_hiddens,
        ffn_num_hiddens=transformer_config.ffn_num_hiddens,
        max_len=transformer_config.max_len,
        dropout=transformer_config.dropout,
        chunk_size=transformer_config.chunk_size
    ).to(device)
    transformer_num_params = sum(p.numel() for p in transformer.parameters())
    transformer_loss_fn = torch.nn.CrossEntropyLoss()
    transformer_optimizer = torch.optim.Adam(transformer.parameters(), lr=3e-4)
    
    print(f"Training Transformer")
    
    t_best_training_loss, t_best_validation_loss, t_best_training_accuracy, t_best_validation_accuracy, \
    t_validation_loss, t_validation_accuracy, t_steps, t_total_epochs, t_time_per_epoch, t_max_memory, \
    t_all_training_losses, t_all_training_accuracies, t_all_validation_losses, t_all_validation_accuracies \
    = train(
        transformer, transformer_train_dataloader, transformer_val_dataloader, train_config.num_epochs, transformer_loss_fn, transformer_optimizer, \
            early_stopping_threshold=train_config.early_stopping_threshold, accumulate_every_i=train_config.accumulate_every_i
        )

    torch.save(transformer, f"{transformer_config.name}.pt")
    
    del transformer
    del transformer_train_dataloader
    del transformer_val_dataloader
    del transformer_dataset
    del transformer_optimizer
    del transformer_loss_fn

    mingru_dataset = torch.load(mingru_dataset_path)
    mingru_train_dataloader, mingru_val_dataloader = get_split(mingru_dataset, batch_size=8)
    
    # Train MinGRU
    mingru = MinGRUSynthetic(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=mingru_config.embedding_dim,
        num_layers=mingru_config.num_layers,
        bidirectional=mingru_config.bidirectional,
        num_classes=train_config.num_classes,
    ).to(device)
    mingru_num_params = sum(p.numel() for p in mingru.parameters())
    mingru_loss_fn = torch.nn.CrossEntropyLoss()
    mingru_optimizer = torch.optim.Adam(mingru.parameters(), lr=3e-4)
    
    print(f"Training MinGRU")
    
    m_best_training_loss, m_best_validation_loss, m_best_training_accuracy, m_best_validation_accuracy, \
    m_validation_loss, m_validation_accuracy, m_steps, m_total_epochs, m_time_per_epoch, m_max_memory, \
    m_all_training_losses, m_all_training_accuracies, m_all_validation_losses, m_all_validation_accuracies \
    = train(
        mingru, mingru_train_dataloader, mingru_val_dataloader, train_config.num_epochs, mingru_loss_fn, mingru_optimizer, \
            early_stopping_threshold=train_config.early_stopping_threshold, accumulate_every_i=train_config.accumulate_every_i
    )

    torch.save(mingru, f"{mingru_config.name}.pt")
    
    del mingru
    del mingru_train_dataloader
    del mingru_val_dataloader
    del mingru_optimizer
    del mingru_loss_fn

    row = get_new_row()
    row['Model'] = transformer_config.name
    row['Layers'] = transformer_config.num_layers
    row['Parameters'] = transformer_num_params
    row['Training Steps'] = t_steps
    row['Number of Epochs'] = t_total_epochs
    row['Time Per Epoch'] = t_time_per_epoch
    row['Best Validation Accuracy'] = t_best_validation_accuracy
    row['Validation Accuracy'] = t_validation_accuracy
    row['Validation Loss'] = t_validation_loss
    row['Best Validation Loss'] = t_best_validation_loss
    row['Best Training Loss'] = t_best_training_loss
    row['Best Training Accuracy'] = t_best_training_accuracy
    row['Max Memory'] = t_max_memory
    row['GPU'] = gpu_name

    append_line(out_path, row)

    row = get_new_row()
    row['Model'] = mingru_config.name
    row['Layers'] = mingru_config.num_layers
    row['Parameters'] = mingru_num_params
    row['Training Steps'] = m_steps
    row['Number of Epochs'] = m_total_epochs
    row['Time Per Epoch'] = m_time_per_epoch
    row['Best Validation Accuracy'] = m_best_validation_accuracy
    row['Validation Accuracy'] = m_validation_accuracy
    row['Validation Loss'] = m_validation_loss
    row['Best Validation Loss'] = m_best_validation_loss
    row['Best Training Loss'] = m_best_training_loss
    row['Best Training Accuracy'] = m_best_training_accuracy
    row['Max Memory'] = m_max_memory
    row['GPU'] = gpu_name

    append_line(out_path, row)

    # Graph results
    mingru_epochs = range(1, m_total_epochs + 1)
    transformer_epochs = range(1, t_total_epochs + 1)

    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(mingru_epochs, m_all_training_accuracies, label="MinGRU Training Accuracy", marker='o')
    plt.plot(transformer_epochs, t_all_training_accuracies, label="Transformer Training Accuracy", marker='o')
    plt.plot(mingru_epochs, m_all_validation_accuracies, label="MinGRU Validation Accuracy", marker='o')
    plt.plot(transformer_epochs, t_all_validation_accuracies, label="Transformer Validation Accuracy", marker='o')
    plt.title(f"Accuracy vs Epochs", fontsize=16)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"accuracy.png")
    plt.close()

    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(mingru_epochs, m_all_training_losses, label="MinGRU Training Loss", marker='o')
    plt.plot(transformer_epochs, t_all_training_losses, label="Transformer Training Loss", marker='o')
    plt.plot(mingru_epochs, m_all_validation_losses, label="MinGRU Validation Loss", marker='o')
    plt.plot(transformer_epochs, t_all_validation_losses, label="Transformer Validation Loss", marker='o')
    plt.title(f"Loss vs Epochs", fontsize=16)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"loss.png")
    plt.close()
    
if __name__ == '__main__':
    main()
