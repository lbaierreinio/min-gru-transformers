import torch
import argparse
from transformers import AutoTokenizer
from datasets.synthetic.utility import generate_dataset8
from datasets.synthetic.TransformerSyntheticDataset import TransformerSyntheticDataset
from datasets.synthetic.MinGRUSyntheticDataset import MinGRUSyntheticDataset

from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """
    Configuration of the experiment.
    """
    min_seq_len: int = None
    max_seq_len: int = 1022
    transformer_max_seq_len: int = 1024
    mingru_max_seq_len: int = 1024
    num_examples: int = 1000
    tokenizer: str = 'bert-base-uncased'
    alpha: int = 4
    beta: int = 2
    k_split: float = None
    k_indicator: float = 0.8


"""
Script to generate and save a synthetic dataset given the current state
of the dataset configuration file.
"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        help='Training dataset path')
    args = parser.parse_args()

    dataset_path = args.dataset_path

    if dataset_path is None:
        raise ValueError("Dataset path must be specified")

    config = DatasetConfig()
    

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
    tokenizer.padding_side = "left"

    grammars = [
        {
            'S': [(0.25, 'A'), (0.25, 'B'), (0.25, 'C'), (0.25, 'D')],
            'A': [(0.25, 'A'), (0.25, 'B'), (0.25, 'C'), (0.25, 'D')],
            'B': [(0.25, 'A'), (0.25, 'B'), (0.25, 'C'), (0.25, 'D')],
            'C': [(0.25, 'A'), (0.25, 'B'), (0.25, 'C'), (0.25, 'D')],
            'D': [(0.25, 'A'), (0.25, 'B'), (0.25, 'C'), (0.25, 'D')],
        },
        {
            'S': [(0.85, 'A'), (0.05, 'B'), (0.05, 'C'), (0.05, 'D')],
            'A': [(0.05, 'A'), (0.85, 'B'), (0.05, 'C'), (0.05, 'D')],
            'B': [(0.05, 'A'), (0.05, 'B'), (0.85, 'C'), (0.05, 'D')],
            'C': [(0.05, 'A'), (0.05, 'B'), (0.05, 'C'), (0.85, 'D')],
            'D': [(0.85, 'A'), (0.05, 'B'), (0.05, 'C'), (0.05, 'D')],
        },
    ]

    examples, labels = generate_dataset8(
        min_seq_len=config.min_seq_len,
        max_seq_len=config.max_seq_len,
        num_examples=config.num_examples,
        alpha=config.alpha,
        beta=config.beta,
        k_split=config.k_split,
        k_indicator=config.k_indicator,
        grammars=grammars,
    )
    
    mingru_dataset = MinGRUSyntheticDataset(examples, labels, tokenizer, max_length=config.transformer_max_seq_len)
    transformer_dataset = TransformerSyntheticDataset(examples, labels, tokenizer, max_length=config.mingru_max_seq_len)

    torch.save(mingru_dataset, f"mingru_{dataset_path}.pt")
    torch.save(transformer_dataset, f"transformer_{dataset_path}.pt")


if __name__ == '__main__':
    main()
