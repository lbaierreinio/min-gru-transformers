import os
import torch
import argparse
from train.utility import evaluate
from torch.utils.data import DataLoader

"""
Script to evaluate a model on a validation dataset.
"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        help='Validation dataset path')
    parser.add_argument('--model_path', type=str,
                        help='Path to load the model from')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_path = args.dataset_path
    model_path = args.model_path

    if not os.path.exists(model_path) or not os.path.exists(dataset_path):
        raise ValueError("Paths must point to a valid file")

    dataset_path = torch.load(dataset_path)
    validation_dataloader = DataLoader(dataset_path, batch_size=8)

    model = torch.load(model_path).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()

    validation_loss, validation_accuracy = evaluate(
        model, validation_dataloader, loss_fn)

    print(f"Total Validation Loss: {validation_loss}")
    print(f"Total Validation Accuracy: {validation_accuracy}")

if __name__ == '__main__':
    main()
