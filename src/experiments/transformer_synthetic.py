import os
import torch
import argparse
from transformers import AutoTokenizer
from datasets.utility import generate_dataset8
from datasets.SyntheticDataset import SyntheticDataset
from models.LongTransformerClassifier import LongTransformerClassifier
from train.utility import train
from datasets.utility import get_split
from utils.utility import get_new_row, create_file, append_line

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, help='Path to load the dataset from or save the dataset to')
    parser.add_argument('--out_path', type=str, help='Path to save the results to')
    args = parser.parse_args()

    dataset_path = args.dataset_path
    out_path = args.out_path
    
    sequence_length = 4096
    num_examples = 400
    batch_size = 256
    num_labels = 4
    replace = True
    num_subsequences = 4
    token_distance = 100
    start = 100
    end = 400
    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=sequence_length)

    if dataset_path is None or dataset_path is None:
        raise ValueError("dataset_path and out_path must be specified")

    if not os.path.exists(out_path):
        create_file(out_path)

    if os.path.exists(dataset_path):
        dataset = torch.load(dataset_path)
    else:
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

        examples, labels = generate_dataset8(
            seq_len=sequence_length, 
            num_examples=num_examples, 
            grammars=grammars, 
            num_labels=num_labels, 
            num_subsequences=num_subsequences, 
            #token_distance=token_distance, 
            #start=start, 
            #end=end, 
            replace=replace
        )

        dataset = SyntheticDataset(examples, labels, tokenizer, sequence_length)
        torch.save(dataset, dataset_path)
    
    # Obtain split
    train_dataloader, val_dataloader, test_dataloader = get_split(dataset)

    # Define model parameters
    vocab_size = tokenizer.vocab_size
    learning_rate = 1e-4
    num_epochs = 10000
    num_layers = 2

    loss_fn = torch.nn.CrossEntropyLoss()

    model = LongTransformerClassifier(
        vocab_size=vocab_size,
        num_heads=2,
        num_layers=num_layers,
        num_classes=4,
        num_hiddens=128,
        chunk_size=128,
        ffn_num_hiddens=512,
        max_len=sequence_length
    ).cuda()
    

    num_parameters = sum(p.numel() for p in model.parameters())

    total_loss, validation_accuracy, steps, total_epochs, avg_time_per_step = train(model, train_dataloader, val_dataloader, num_epochs, loss_fn, learning_rate, early_stopping=True)

    next_row = get_new_row()

    # Create the new row and update the fields for MinGRU
    next_row['Model'] = 'Transformer'
    next_row['Layers'] = num_layers
    next_row['Parameters'] = num_parameters
    next_row['Sequence Length'] = sequence_length
    next_row['Dataset Size'] = len(dataset)
    next_row['Token Distance'] = token_distance
    next_row['Start'] = start
    next_row['End'] = end
    next_row['Training Steps'] = steps
    next_row['Number of Epochs'] = total_epochs
    next_row['Training Time'] = avg_time_per_step
    next_row['Memory Per Epoch'] = 'TODO'
    next_row['Validation Accuracy'] = validation_accuracy
    next_row['Validation Loss'] = total_loss

    append_line(out_path, next_row)

if __name__ == '__main__':
    main()