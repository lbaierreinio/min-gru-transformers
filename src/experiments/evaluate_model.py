import os
import torch
import argparse
from train.utility import evaluate
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation_dataset_path', type=str,
                        help='Validation dataset path')
    parser.add_argument('--model_in_path', type=str,
                        help='Path to load the model from')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    validation_dataset_path = args.validation_dataset_path
    model_in_path = args.model_in_path

    if not os.path.exists(model_in_path) or not os.path.exists(validation_dataset_path):
        raise ValueError("Paths must point to a valid file")

    validation_dataset_path = torch.load(validation_dataset_path)
    validation_dataloader = DataLoader(validation_dataset_path, batch_size=32)

    model = torch.load(model_in_path).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()

    validation_loss, validation_accuracy = evaluate(
        model, validation_dataloader, loss_fn, evaluation_type='Validation')

    print(f"Total Validation Loss: {round(validation_loss, 2)}")
    print(f"Total Validation Accuracy: {round(validation_accuracy, 2)}")
    print(f"Averaged Validation Loss: {round(validation_loss/len(validation_dataloader.dataset), 2)}")

if __name__ == '__main__':
    main()
