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
            'S': [(0.95, 'A'), (0.05, 'B')],
            'A': [(0.95, 'A'), (0.05, 'B')],
            'B': [(0.95, 'A'), (0.05, 'B')],
        },
        {
            'S': [(0.50, 'B'), (0.50, 'A')],
            'B': [(0.50, 'B'), (0.50, 'A')],
            'A': [(0.50, 'B'), (0.50, 'A')],
        },
        {
            'S': [(0.75, 'A'), (0.25, 'C')],
            'A': [(0.75, 'A'), (0.25, 'C')],
            'C': [(0.75, 'A'), (0.25, 'C')],
        }
    ]

    orders = [
        [0, 1, 2, 0],
        [1, 2, 0, 0],
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
        even=config.even
    )

    dataset = SyntheticDataset(
        examples, labels, tokenizer, config.sequence_length)

    torch.save(dataset, dataset_path)


if __name__ == '__main__':
    main()
