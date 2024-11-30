import os
import torch
import argparse
from transformers import AutoTokenizer
from datasets.utility import get_split
from train.utility import train, evaluate
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
    parser.add_argument('--validation_dataset_path', type=str,
                        help='Validation dataset path')
    parser.add_argument('--out_path', type=str,
                        help='Path to save the results to')
    parser.add_argument('--model', type=int,
                        help='Model to use: [0: MinGRU, 1: Transformer]')

    args = parser.parse_args()

    train_dataset_path = args.train_dataset_path
    validation_dataset_path = args.validation_dataset_path
    out_path = args.out_path
    model = args.model

    if not os.path.exists(out_path):
        create_file(out_path)

    if train_dataset_path is None or validation_dataset_path is None:
        raise ValueError("Paths must be specified")

    if model not in [0, 1]:
        raise ValueError("Model must be 0 or 1")

    if not os.path.exists(train_dataset_path) or not os.path.exists(validation_dataset_path):
        raise ValueError("Paths must point to a valid file")

    # (2) Load Dataset
    dataset_config = DatasetConfig()

    tokenizer = AutoTokenizer.from_pretrained(
        dataset_config.tokenizer, model_max_length=dataset_config.sequence_length)

    train_dataset = torch.load(train_dataset_path)
    val_dataset = torch.load(validation_dataset_path)
    train_dataloader1, val_dataloader1, _ = get_split(train_dataset)
    _, val_dataloader2, _ = get_split(val_dataset)

    # (3) Load Training Parameters
    train_config = TrainConfig()
    loss_fn = torch.nn.CrossEntropyLoss()

    # (4) Define Model and Configuration
    mingru_config = MinGRUConfig()
    transformer_config = TransformerConfig()
    vocab_size = tokenizer.vocab_size

    config_dict = {
        0: mingru_config,
        1: transformer_config
    }

    model_dict = {
        0: MinGRUClassifier(
            vocab_size=vocab_size,
            embedding_dim=mingru_config.embedding_dim,
            expansion_factor=mingru_config.expansion_factor,
            num_layers=mingru_config.num_layers,
            bidirectional=mingru_config.bidirectional,
            num_classes=mingru_config.labels
        ),
        1: LongTransformerClassifier(
            vocab_size=vocab_size,
            num_heads=transformer_config.num_heads,
            num_layers=transformer_config.num_layers,
            num_classes=dataset_config.num_labels,
            num_hiddens=transformer_config.num_hiddens,
            ffn_num_hiddens=transformer_config.ffn_num_hiddens,
            chunk_size=transformer_config.chunk_size,
            max_len=dataset_config.sequence_length,
        )
    }

    config = config_dict[model]
    model = model_dict[model].cuda()
    num_parameters = sum(p.numel() for p in model.parameters())

    # (5) Train Model
    _, _, steps, total_epochs, avg_time_per_step = train(
        model, train_dataloader1, val_dataloader1, train_config.num_epochs, loss_fn, train_config.learning_rate, early_stopping=train_config.early_stopping)

    validation_accuracy, total_loss = evaluate(model, val_dataloader2, loss_fn)

    # (6) Store Results
    next_row = get_new_row()

    # Create the new row and update the fields for MinGRU
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
