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
    sequence_length: int = 128
    cls_tokens: int = 4
    num_examples: int = 2000
    tokenizer: str = 'bert-base-uncased'
    alpha: int = 1
    beta: int = 2
    k_split: float = 0.02
    k_indicator: float = 0.1
    insert_cls_tokens: bool = True
    chunk_size: int = 32

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
    

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer, model_max_length=config.sequence_length)
    tokenizer.padding_side = "left"

    grammars = [
        {
            'S': [(0.94, 'A'), (0.02, 'B'), (0.02, 'C'), (0.02, 'D')],
            'A': [(0.94, 'A'), (0.02, 'B'), (0.02, 'C'), (0.02, 'D')],
            'B': [(0.94, 'A'), (0.02, 'B'), (0.02, 'C'), (0.02, 'D')],
            'C': [(0.94, 'A'), (0.02, 'B'), (0.02, 'C'), (0.02, 'D')],
            'D': [(0.94, 'A'), (0.02, 'B'), (0.02, 'C'), (0.02, 'D')],
        },
        {
            'S': [(0.25, 'A'), (0.25, 'B'), (0.25, 'C'), (0.25, 'D')],
            'A': [(0.02, 'A'), (0.94, 'B'), (0.02, 'C'), (0.02, 'D')],
            'B': [(0.02, 'A'), (0.02, 'B'), (0.94, 'C'), (0.02, 'D')],
            'C': [(0.02, 'A'), (0.02, 'B'), (0.02, 'C'), (0.94, 'D')],
            'D': [(0.94, 'A'), (0.02, 'B'), (0.02, 'C'), (0.02, 'D')],
        },
    ]

    assert config.sequence_length % config.chunk_size == 0
    num_chunks = int(config.sequence_length // config.chunk_size)

    examples, labels = generate_dataset8(
        seq_len=config.sequence_length - num_chunks,
        num_examples=config.num_examples,
        alpha=config.alpha,
        beta=config.beta,
        k_split=config.k_split,
        k_indicator=config.k_indicator,
        grammars=grammars,
    )

    transformer_dataset = TransformerSyntheticDataset(
        examples, labels, tokenizer, config.sequence_length, chunk_size=32)
    
    mingru_dataset = MinGRUSyntheticDataset(
        examples, labels, tokenizer, config.sequence_length+2
    )

    torch.save(transformer_dataset, f"transformer_{dataset_path}.pt")
    torch.save(mingru_dataset, f"mingru_{dataset_path}.pt")


if __name__ == '__main__':
    main()
