import os
import torch
import argparse
from transformers import AutoTokenizer
from datasets.utility import get_split
from train.utility import train
from experiments.dataset_config import DatasetConfig
from experiments.train_config import TrainConfig
from experiments.mingru_config import MinGRUConfig
from experiments.transformer_config import TransformerConfig
from models.MinGRUClassifier import MinGRUClassifier
from models.LongTransformerClassifier import LongTransformerClassifier
from utils.utility import get_new_row, create_file, append_line


def main():
    # (1) Retrieve arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset_path', type=str,
                        help='Training dataset path')
    parser.add_argument('--out_path', type=str,
                        help='Path to save the results to')
    parser.add_argument('--model_out_path', type=str,
                        help='Path to save the model to')
    parser.add_argument('--model', type=int,
                        help='Model to use: [0: MinGRU, 1: Transformer]')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset_path = args.train_dataset_path
    out_path = args.out_path
    model_out_path = args.model_out_path
    model = args.model

    persist_results = False

    if out_path is not None and not os.path.exists(out_path):
        create_file(out_path)

    if train_dataset_path is None:
        raise ValueError("Paths must be specified")

    if model not in [0, 1]:
        raise ValueError("Model must be 0 or 1")

    if not os.path.exists(train_dataset_path):
        raise ValueError("Path must point to a valid file")

    # (2) Load Dataset
    dataset_config = DatasetConfig()

    tokenizer = AutoTokenizer.from_pretrained(
        dataset_config.tokenizer, model_max_length=dataset_config.sequence_length)

    train_dataset = torch.load(train_dataset_path)
    train_dataloader, val_dataloader = get_split(train_dataset)

    # (3) Load Training Parameters
    train_config = TrainConfig()
    loss_fn = torch.nn.CrossEntropyLoss()

    # (4) Define Model and Configuration
    vocab_size = tokenizer.vocab_size

    if model == 0:
        config = MinGRUConfig()
        model = MinGRUClassifier(
            vocab_size=vocab_size,
            embedding_dim=config.embedding_dim,
            expansion_factor=config.expansion_factor,
            num_layers=config.num_layers,
            bidirectional=config.bidirectional,
            num_classes=dataset_config.num_labels,
        ).cuda()
    else:
        config = TransformerConfig()
        model = LongTransformerClassifier(
            vocab_size=vocab_size,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            num_classes=dataset_config.num_labels,
            num_hiddens=config.num_hiddens,
            ffn_num_hiddens=config.ffn_num_hiddens,
            chunk_size=config.chunk_size,
            max_len=dataset_config.sequence_length,
        ).to(device)

    num_parameters = sum(p.numel() for p in model.parameters())

    # (5) Train Model
    validation_accuracy, total_loss, steps, total_epochs, avg_time_per_step = train(
        model, train_dataloader, val_dataloader, train_config.num_epochs, loss_fn, train_config.learning_rate, early_stopping=train_config.early_stopping)

    torch.save(model, config.name + model_out_path)

    # (6) Store Results

    if persist_results:
        # Create the new row and update the fields for MinGRU
        next_row = get_new_row()
        next_row['Model'] = config.name
        next_row['Layers'] = config.num_layers
        next_row['Parameters'] = num_parameters
        next_row['Sequence Length'] = dataset_config.sequence_length
        next_row['Dataset Size'] = len(train_dataset)
        next_row['Token Distance'] = 'N/A'
        next_row['Start'] = dataset_config.start
        next_row['End'] = dataset_config.end
        next_row['Training Steps'] = steps
        next_row['Number of Epochs'] = total_epochs
        next_row['Training Time'] = avg_time_per_step
        next_row['Memory Per Epoch'] = 'TODO'
        next_row['Validation Accuracy'] = validation_accuracy
        next_row['Validation Loss'] = total_loss

        append_line(out_path, next_row)


if __name__ == '__main__':
    main()