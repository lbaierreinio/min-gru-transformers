import torch
import numpy as np
import torch.profiler
from train.utility import profile_inference
from dataclasses import dataclass
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from models.MinGRUSynthetic import MinGRUSynthetic
from models.TransformerSynthetic import TransformerSynthetic

@dataclass
class MinGRUConfig:
    """
    Configuration for minGRU model.
    """
    name: str = 'mingru'
    num_layers: int = 6
    embedding_dim: int = 384
    bidirectional: bool = False

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
class ProfileInferenceConfig:
    """
    Configuration for profiling.
    """
    start_length: int = 2048
    end_length: int = 4096
    step: int = 512
    num_classes: int = 8

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    mingru_config = MinGRUConfig()
    transformer_config = TransformerConfig()
    profile_config = ProfileInferenceConfig()

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

    lengths = [i for i in range(profile_config.start_length, profile_config.end_length, profile_config.step)]

    all_min_gru_parallel_mean_memory = []
    all_min_gru_sequential_mean_memory = []
    all_min_gru_parallel_mean_time = []
    all_min_gru_sequential_mean_time = []
    all_min_gru_parallel_std_memory = []
    all_min_gru_sequential_std_memory = []
    all_min_gru_parallel_std_time = []
    all_min_gru_sequential_std_time = []

    for seq_len in lengths:
        print(f"Profiling sequence length: {seq_len}")
        min_gru_parallel_mean_memory, min_gru_parallel_mean_time, min_gru_parallel_std_memory, min_gru_parallel_std_time = \
            profile_inference(mingru, seq_len, tokenizer.vocab_size, device, profile_steps = 1, warmup_steps=1, is_sequential=False)
        min_gru_sequential_mean_memory, min_gru_sequential_mean_time, min_gru_sequential_std_memory, min_gru_sequential_std_time = \
            profile_inference(mingru, seq_len, tokenizer.vocab_size, device, profile_steps = 1, warmup_steps=1, is_sequential=True)
        
        # Add values
        all_min_gru_parallel_mean_memory.append(min_gru_parallel_mean_memory)
        all_min_gru_sequential_mean_memory.append(min_gru_sequential_mean_memory)
        all_min_gru_parallel_mean_time.append(min_gru_parallel_mean_time)
        all_min_gru_sequential_mean_time.append(min_gru_sequential_mean_time)
        all_min_gru_parallel_std_memory.append(min_gru_parallel_std_memory)
        all_min_gru_sequential_std_memory.append(min_gru_sequential_std_memory)
        all_min_gru_parallel_std_time.append(min_gru_parallel_std_time)
        all_min_gru_sequential_std_time.append(min_gru_sequential_std_time)

    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    
    # Get parameters
    num_parameters = sum(p.numel() for p in mingru.parameters())
    print(f"MinGRU Parameters: {num_parameters}")

    # Plot time per epoch with error bars for standard deviation
    plt.figure(figsize=(10, 5))
    plt.errorbar(
        lengths, all_min_gru_parallel_mean_time, yerr=np.array(all_min_gru_parallel_std_time),
        label="MinGRU Parallel Inference", marker='o', capsize=5, linestyle='-', ecolor='red'
    )
    plt.errorbar(
        lengths, all_min_gru_sequential_mean_time, yerr=np.array(all_min_gru_sequential_std_time),
        label="MinGRU Sequential Inference", marker='o', capsize=5, linestyle='-', ecolor='blue'
    )
    plt.title(f"Time per Epoch vs. Sequence Length on {gpu_name}", fontsize=16)
    plt.xlabel("Sequence Length", fontsize=14)
    plt.ylabel("Time per Epoch (s)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"time_inference.png")
    plt.close()

    # Plot memory usage with error bars for standard deviation
    plt.figure(figsize=(10, 5))
    plt.errorbar(
        lengths, all_min_gru_parallel_mean_memory, yerr=np.array(all_min_gru_parallel_std_memory),
        label="MinGRU Parallel Inference", marker='o', capsize=5, linestyle='-', ecolor='red'
    )
    plt.errorbar(
        lengths, all_min_gru_sequential_mean_memory, yerr=np.array(all_min_gru_sequential_std_memory),
        label="MinGRU Sequential Inference", marker='o', capsize=5, linestyle='-', ecolor='blue'
    )
    plt.title(f"Memory vs. Sequence Length on {gpu_name}", fontsize=16)
    plt.xlabel("Sequence Length", fontsize=14)
    plt.ylabel("Memory Usage (MB)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"memory_inference.png")
    plt.close()


if __name__ == '__main__':
    main()
