import os
import torch
import argparse
from transformers import AutoTokenizer
from datasets.utility import generate_dataset8
from datasets.SyntheticDataset import SyntheticDataset

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str, help='Path to load the dataset from or save the dataset to')
    args = parser.parse_args()

    path = args.dataset_path
    if path is None:
        raise ValueError("Please provide a dataset")

    if os.path.exists(path):
        dataset = torch.load(path)

    else:
        num_examples = 5
        sequence_length = 1024
        batch_size = 256
        num_labels = 4
        replace = True
        num_subsequences = 4
        token_distance = 3
        start = 1020
        end = 1024

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

        model_name = 'bert-base-uncased'

        tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=sequence_length)

        examples, labels = generate_dataset8(
            seq_len=sequence_length, 
            num_examples=num_examples, 
            grammars=grammars, 
            num_labels=num_labels, 
            num_subsequences=num_subsequences, 
            token_distance=token_distance, 
            start=start, 
            end=end, 
            replace=replace
        )

        dataset = SyntheticDataset(examples, labels, tokenizer, sequence_length)
        torch.save(dataset, path)


if __name__ == '__main__':
    main()