import os
import torch
import pandas as pd
import torch.nn.functional as F

def create_file(path):
    data = []
    # adding header
    headerList = ['Model', 'Layers', 'Parameters', 'Sequence Length', 'Dataset Size', 'Token Distance', 'Start', 'End', 'Training Steps', 'Number of Epochs', 'Training Time', 'Memory Per Epoch', 'Validation Accuracy', 'Validation Loss']

    # Convert the data to a DataFrame
    df = pd.DataFrame(data, columns=headerList)

    # Write the DataFrame to a CSV file
    df.to_csv(path, index=False)

def append_line(path, data):
    assert os.path.exists(path), f"File does not exist at {path}"
    new_row = pd.DataFrame([data])
    new_row.to_csv(path, mode='a', index=False, header=False)

def get_new_row():
    new_row = {
        'Model': 'GPT-4',
        'Layers': 24,
        'Parameters': '1.7B',
        'Sequence Length': 1024,
        'Dataset Size': 10000,
        'Token Distance': 3,
        'Start': 1020,
        'End': 1024,
        'Training Steps': 1,
        'Number of Epochs': 1,
        'Training Time': 120,
        'Memory Per Epoch': '10GB',
        'Validation Accuracy': 0.9,
        'Validation Loss': 0.1
    }
    return new_row

