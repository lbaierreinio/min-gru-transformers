import os
import pandas as pd


def create_file(path):
    data = []
    # adding header
    headerList = ['Model', 'Layers', 'Parameters', 'Sequence Length', 'Dataset Size', 'Training Steps',
                  'Number of Epochs', 'Training Time', 'Memory Per Epoch', 'Validation Accuracy', 'Validation Loss']

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
        'Sequence Length': None,
        'Dataset Size': None,
        'Training Steps': None,
        'Number of Epochs': None,
        'Training Time': None,
        'Memory Per Epoch': None,
        'Validation Accuracy': None,
        'Validation Loss': None
    }
    return new_row
