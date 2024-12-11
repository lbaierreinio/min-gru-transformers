import torch
import numpy as np
import torch.profiler
from train.utility import profile_train
from dataclasses import dataclass
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets.synthetic.utility import generate_dataset8
from models.MinGRUSynthetic import MinGRUSynthetic
from models.TransformerSynthetic import TransformerSynthetic
from datasets.synthetic.TransformerSyntheticDataset import TransformerSyntheticDataset

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
    num_classes = 4
    ffn_num_hiddens = 1024
    dropout = 0.1
    max_len: int = 4096
    chunk_size: int = 512

@dataclass
class ProfileTrainConfig:
    """
    Configuration for profiling.
    """
    batch_size: int = 4 # Small batch size to avoid OOM for large sequence lengths
    dataset_size: int = 1000
    num_classes: int = 4
    start_length: int = 512
    end_length: int = 4096
    step: int = 512

"""
Script to profile the training of two models on a synthetic dataset as the sequence length grows.
"""
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    mingru_config = MinGRUConfig()
    transformer_config = TransformerConfig()
    profile_config = ProfileTrainConfig()

    mingru = MinGRUSynthetic(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=mingru_config.embedding_dim,
        num_layers=mingru_config.num_layers,
        bidirectional=mingru_config.bidirectional,
        num_classes=profile_config.num_classes
    ).to(device)

    transformer = TransformerSynthetic(
        vocab_size=tokenizer.vocab_size,
        num_heads=transformer_config.num_heads,
        num_layers=transformer_config.num_layers,
        num_classes=transformer_config.num_classes,
        num_hiddens=transformer_config.num_hiddens,
        ffn_num_hiddens=transformer_config.ffn_num_hiddens,
        max_len=transformer_config.max_len,
        dropout=transformer_config.dropout,
        chunk_size=transformer_config.chunk_size
    ).to(device)

    optimizer = torch.optim.Adam(mingru.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    lengths = [i for i in range(profile_config.start_length, profile_config.end_length, profile_config.step)]

    grammars = [
        {
            'S': [(1.0, 'A')],
            'A': [(1.0, 'A')],
        },
        {
            'S': [(1.0, 'B')],
            'B': [(1.0, 'B')],
        },
    ]

    all_min_gru_mean_memory = []
    all_transformer_mean_memory = []
    all_min_gru_mean_time = []
    all_transformer_mean_time = []
    all_min_gru_std_memory = []
    all_transformer_std_memory = []
    all_min_gru_std_time = []
    all_transformer_std_time = []

    for seq_len in lengths:
        examples, labels = generate_dataset8(
            min_seq_len=None,
            max_seq_len=seq_len,
            num_examples=profile_config.dataset_size,
            alpha=2,
            beta=4,
            k_split=None,
            k_indicator=None,
            grammars=grammars
        )              
        examples = examples
        labels = labels

        mingru.train()
        transformer.train()
        
        dataset =  TransformerSyntheticDataset(examples, labels, tokenizer, max_length=seq_len)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        min_gru_mean_memory, min_gru_mean_time, min_gru_std_memory, min_gru_std_time = profile_train(dataloader, device, mingru, loss_fn, optimizer)
        transformer_mean_memory, transformer_mean_time, transformer_std_memory, transformer_std_time = profile_train(dataloader, device, transformer, loss_fn, optimizer)
        
        # Add values
        all_min_gru_mean_memory.append(min_gru_mean_memory)
        all_transformer_mean_memory.append(transformer_mean_memory)
        all_min_gru_mean_time.append(min_gru_mean_time)
        all_transformer_mean_time.append(transformer_mean_time)
        all_min_gru_std_memory.append(min_gru_std_memory)
        all_transformer_std_memory.append(transformer_std_memory)
        all_min_gru_std_time.append(min_gru_std_time)
        all_transformer_std_time.append(transformer_std_time)

    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    # Plot time per epoch with error bars for standard deviation
    plt.figure(figsize=(10, 5))
    plt.errorbar(
        lengths, all_min_gru_mean_time, yerr=np.array(all_min_gru_std_time),
        label="MinGRU", marker='o', capsize=5, linestyle='-', ecolor='red'
    )
    plt.errorbar(
        lengths, all_transformer_mean_time, yerr=np.array(all_transformer_std_time),
        label="Transformer", marker='o', capsize=5, linestyle='-', ecolor='blue'
    )
    plt.title(f"Time per Epoch vs. Sequence Length on {gpu_name}", fontsize=16)
    plt.xlabel("Sequence Length", fontsize=14)
    plt.ylabel("Time per Epoch (s)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"time_epochs.png")
    plt.close()

    # Plot memory usage with error bars for standard deviation
    plt.figure(figsize=(10, 5))
    plt.errorbar(
        lengths, all_min_gru_mean_memory, yerr=np.array(all_min_gru_std_memory),
        label="MinGRU", marker='o', capsize=5, linestyle='-', ecolor='red'
    )
    plt.errorbar(
        lengths, all_transformer_mean_memory, yerr=np.array(all_transformer_std_memory),
        label="Transformer", marker='o', capsize=5, linestyle='-', ecolor='blue'
    )
    plt.title(f"Memory vs. Sequence Length on {gpu_name}", fontsize=16)
    plt.xlabel("Sequence Length", fontsize=14)
    plt.ylabel("Memory Usage (MB)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"memory_epochs.png")
    plt.close()


if __name__ == '__main__':
    main()
