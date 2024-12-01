import torch
import argparse
from transformers import AutoTokenizer
from datasets.utility import generate_dataset8
from datasets.SyntheticDataset import SyntheticDataset
from experiments.dataset_config import DatasetConfig

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

    grammars = [
        {
            'S': [(0.25, 'A'), (0.25, 'B'), (0.25, 'C'), (0.25, 'D')],
            'A': [(0.25, 'A'), (0.25, 'B'), (0.25, 'C'), (0.25, 'D')],
            'B': [(0.25, 'A'), (0.25, 'B'), (0.25, 'C'), (0.25, 'D')],
            'C': [(0.25, 'A'), (0.25, 'B'), (0.25, 'C'), (0.25, 'D')],
            'D': [(0.25, 'A'), (0.25, 'B'), (0.25, 'C'), (0.25, 'D')],
        },
        {
            'S': [(0.25, 'A'), (0.25, 'B'), (0.25, 'C'), (0.25, 'D')],
            'A': [(0.05, 'A'), (0.85, 'B'), (0.05, 'C'), (0.05, 'D')],
            'B': [(0.05, 'A'), (0.05, 'B'), (0.85, 'C'), (0.05, 'D')],
            'C': [(0.05, 'A'), (0.05, 'B'), (0.05, 'C'), (0.85, 'D')],
            'D': [(0.85, 'A'), (0.05, 'B'), (0.05, 'C'), (0.05, 'D')],
        },
    ]

    orders = [
        [0, 1],
        [1, 0],
    ]

    examples, labels = generate_dataset8(
        seq_len=config.sequence_length,
        num_examples=config.num_examples,
        grammars=grammars,
        num_labels=config.num_labels,
        num_subsequences=config.num_subsequences,
        start=config.start,
        end=config.end,
        orders=orders,
    )

    dataset = SyntheticDataset(
        examples, labels, tokenizer, config.sequence_length)

    torch.save(dataset, dataset_path)


if __name__ == '__main__':
    main()
