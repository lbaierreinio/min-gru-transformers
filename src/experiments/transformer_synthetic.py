import os
import torch
import argparse
from transformers import AutoTokenizer
from datasets.utility import get_split
from train.utility import train, evaluate
from experiments.dataset_config import DatasetConfig
from models.LongTransformerClassifier import LongTransformerClassifier
from utils.utility import get_new_row, create_file, append_line


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset_path', type=str,
                        help='Training dataset path')
    parser.add_argument('--validation_dataset_path', type=str,
                        help='Validation dataset path')
    parser.add_argument('--out_path', type=str,
                        help='Path to save the results to')

    args = parser.parse_args()

    train_dataset_path = args.train_dataset_path
    validation_dataset_path = args.validation_dataset_path
    out_path = args.out_path

    if not os.path.exists(out_path):
        create_file(out_path)

    if train_dataset_path is None or validation_dataset_path is None:
        raise ValueError("Paths must be specified")

    config = DatasetConfig()

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer, model_max_length=config.sequence_length)

    train_dataset = torch.load(train_dataset_path)
    val_dataset = torch.load(validation_dataset_path)
    train_dataloader1, val_dataloader1, _ = get_split(train_dataset)
    _, val_dataloader2, _ = get_split(val_dataset)

    # Define model parameters
    vocab_size = tokenizer.vocab_size
    learning_rate = 1e-4
    num_epochs = 10000
    num_layers = 4

    loss_fn = torch.nn.CrossEntropyLoss()

    model = LongTransformerClassifier(
        vocab_size=vocab_size,
        num_heads=8,
        num_layers=num_layers,
        num_classes=4,
        num_hiddens=128,
        ffn_num_hiddens=2048
    ).cuda()

    num_parameters = sum(p.numel() for p in model.parameters())

    _, _, steps, total_epochs, avg_time_per_step = train(
        model, train_dataloader1, val_dataloader1, num_epochs, loss_fn, learning_rate, early_stopping=True)

    validation_accuracy, total_loss = evaluate(model, val_dataloader2, loss_fn)

    next_row = get_new_row()

    # Create the new row and update the fields for MinGRU
    next_row['Model'] = 'Transformer'
    next_row['Layers'] = num_layers
    next_row['Parameters'] = num_parameters
    next_row['Sequence Length'] = config.sequence_length
    next_row['Dataset Size'] = len(dataset1)
    next_row['Token Distance'] = 'N/A'
    next_row['Start'] = config.start
    next_row['End'] = config.end
    next_row['Training Steps'] = steps
    next_row['Number of Epochs'] = total_epochs
    next_row['Training Time'] = avg_time_per_step
    next_row['Memory Per Epoch'] = 'TODO'
    next_row['Validation Accuracy'] = validation_accuracy
    next_row['Validation Loss'] = total_loss

    append_line(out_path, next_row)


if __name__ == '__main__':
    main()
