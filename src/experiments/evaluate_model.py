import os
import torch
import argparse
from train.utility import evaluate
from torch.utils.data import DataLoader
from experiments.dataset_config import DatasetConfig
from utils.utility import get_new_row, create_file, append_line


def main():
    # (1) Retrieve arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation_dataset_path', type=str,
                        help='Validation dataset path')
    parser.add_argument('--model_in_path', type=str,
                        help='Path to load the model from')
    parser.add_argument('--out_path', type=str,
                        help='Path to save the results to')

    args = parser.parse_args()

    validation_dataset_path = args.validation_dataset_path
    model_in_path = args.model_in_path
    out_path = args.out_path

    if not os.path.exists(out_path):
        create_file(out_path)

    if not os.path.exists(model_in_path) or not os.path.exists(validation_dataset_path):
        raise ValueError("Paths must point to a valid file")

    # (2) Load Dataset
    dataset_config = DatasetConfig()

    train_dataset = torch.load(validation_dataset_path)
    train_dataloader = DataLoader(train_dataset, batch_size=32)

    # (4) Define Model and Configuration

    model = torch.load(model_in_path).cuda()
    loss_fn = torch.nn.CrossEntropyLoss()

    num_parameters = sum(p.numel() for p in model.parameters())

    total_loss, validation_accuracy = evaluate(
        model, train_dataloader, loss_fn, evaluation_type='Validation')

    # (6) Store Results
    next_row = get_new_row()

    # Create the new row and update the fields for MinGRU
    next_row['Model'] = 'N/A'
    next_row['Layers'] = 'N/A'
    next_row['Parameters'] = num_parameters
    next_row['Sequence Length'] = dataset_config.sequence_length
    next_row['Dataset Size'] = dataset_config.num_examples
    next_row['Training Steps'] = 'N/A'
    next_row['Number of Epochs'] = 'N/A'
    next_row['Training Time'] = 'N/A'
    next_row['Memory Per Epoch'] = 'TODO'
    next_row['Validation Accuracy'] = validation_accuracy
    next_row['Validation Loss'] = total_loss

    append_line(out_path, next_row)


if __name__ == '__main__':
    main()
