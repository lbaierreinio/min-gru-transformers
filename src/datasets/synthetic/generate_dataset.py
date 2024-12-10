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
    max_seq_len: int = 2048
    num_examples: int = 4000
    tokenizer: str = 'bert-base-uncased'
    alpha: int = 4
    beta: int = 2
    k_split: float = 0.05
    k_indicator: float = 0.025


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
            'S': [(0.90, 'A'), (0.05, 'B'), (0.05, 'C')],
            'A': [(0.90, 'A'), (0.05, 'B'), (0.05, 'C')],
            'B': [(0.90, 'A'), (0.05, 'B'), (0.05, 'C')],
            'C': [(0.90, 'A'), (0.05, 'B'), (0.05, 'C')],
        },
        {
            'S': [(0.05, 'A'), (0.90, 'B'), (0.05, 'C')],
            'A': [(0.05, 'A'), (0.90, 'B'), (0.05, 'C')],
            'B': [(0.05, 'A'), (0.90, 'B'), (0.05, 'C')],
            'C': [(0.05, 'A'), (0.90, 'B'), (0.05, 'C')],
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
    
    mingru_dataset = MinGRUSyntheticDataset(examples, labels, tokenizer, max_length=config.max_seq_len+2)
    transformer_dataset = TransformerSyntheticDataset(examples, labels, tokenizer, max_length=config.max_seq_len)

    torch.save(mingru_dataset, f"mingru_{dataset_path}.pt")
    torch.save(transformer_dataset, f"transformer_{dataset_path}.pt")


if __name__ == '__main__':
    main()
