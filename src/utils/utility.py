import os
import pandas as pd


def create_file(path):
    data = []
    # adding header
    headerList = ['Model', 'Layers', 'Parameters', 'Dataset Path', 'Training Steps',
                  'Number of Epochs', 'Time Per Epoch', 'Validation Accuracy', 'Best Validation Accuracy', 'Validation Loss', 'Best Validation Loss', 'Best Training Loss', 'Best Training Accuracy', 'Max Memory', 'GPU']

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
        'Model': None,
        'Layers': None,
        'Parameters': None,
        'Dataset Path': None,
        'Training Steps': None,
        'Number of Epochs': None,
        'Time Per Epoch': None,
        'Validation Accuracy': None,
        'Best Validation Accuracy': None,
        'Validation Loss': None,
        'Best Validation Loss': None,
        'Best Training Loss': None,
        'Best Training Accuracy': None,
        'Max Memory': None,
        'GPU': None
    }
    return new_row
