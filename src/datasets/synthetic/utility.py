import torch
import numpy as np
from torch.utils.data import DataLoader

EASY_GRAMMARS = [
        {
            'S': [(0.90, 'A'), (0.05, 'B'), (0.05, 'C')],
            'A': [(0.90, 'A'), (0.05, 'B'), (0.05, 'C')],
            'B': [(0.90, 'A'), (0.05, 'B'), (0.05, 'C')],
            'C': [(0.90, 'A'), (0.05, 'B'), (0.05, 'C')],
        },
        {
            'S': [(0.05, 'A'), (0.90, 'B'), (0.05, 'C')],
            'A': [(0.05, 'A'), (0.90, 'B'), (0.05, 'C')],
            'B': [(0.05, 'A'), (0.90, 'B'), (0.05, 'C')],
            'C': [(0.05, 'A'), (0.90, 'B'), (0.05, 'C')],
        },
       
    ]

DIFFICULT_GRAMMARS = [
        {
            'S': [(0.25, 'A'), (0.25, 'B'), (0.25, 'C'), (0.25, 'D')],
            'A': [(0.25, 'A'), (0.25, 'B'), (0.25, 'C'), (0.25, 'D')],
            'B': [(0.25, 'A'), (0.25, 'B'), (0.25, 'C'), (0.25, 'D')],
            'C': [(0.25, 'A'), (0.25, 'B'), (0.25, 'C'), (0.25, 'D')],
            'D': [(0.25, 'A'), (0.25, 'B'), (0.25, 'C'), (0.25, 'D')],
        },
        {
            'S': [(0.85, 'A'), (0.05, 'B'), (0.05, 'C'), (0.05, 'D')],
            'A': [(0.05, 'A'), (0.85, 'B'), (0.05, 'C'), (0.05, 'D')],
            'B': [(0.05, 'A'), (0.05, 'B'), (0.85, 'C'), (0.05, 'D')],
            'C': [(0.05, 'A'), (0.05, 'B'), (0.05, 'C'), (0.85, 'D')],
            'D': [(0.85, 'A'), (0.05, 'B'), (0.05, 'C'), (0.05, 'D')],
        },
    ]

def generate_grammar(rules, seq_len):
    """
    Generate a sequence of tokens based on the provided rules.

    Args:
        rules: dict 
            Dictionary of rules for the grammar.
        seq_len: int 
            Length of the sequence to be generated.
    
    Returns:
        sequence: list (List of tokens generated based on the rules.)
    """
    sequence = []
    state = 'S'

    while len(sequence) < seq_len:
        current_rules = rules[state]
        next_state_idx = np.random.choice(
            np.arange(len(current_rules)), p=list(map(lambda x: x[0], current_rules)))
        state = current_rules[next_state_idx][1]
        sequence.append(state)

    return sequence

def generate_dataset8(*, min_seq_len=None, max_seq_len, num_examples, grammars, alpha, beta, k_split=None, k_indicator=None):
    """
    Long-Context Task for both Summarization & Long Range Dependencies. Specifically,
    the model is asked to solve two tasks simultaneously: 
    (a) Recall which order the two grammars appeared in.
    (b) Determine if the two indicator tokens are the same.

    Args:
        min_seq_len: int 
            Minimum sequence length of the dataset. Dataset size will have fixed lenth if this argument is not provided.
        max_seq_len: int 
            Maximum sequence length of the dataset.
        num_examples: int 
            Number of examples in the dataset.
        grammars: list 
            List of 2 grammars to be used in the dataset.
        alpha: int 
            Alpha parameter for Beta distribution of the sequence length.
        beta: int 
            Beta parameter for Beta distribution of the sequence length.
        k_split: float 
            Standard deviation for the split of the two subsequences. Split is in the middle of the sequence if this argument is not provided.
        k_indicator: float 
            Standard deviation for the indicator tokens. No indicator tokens are inserted if this argument is not provided.
    Returns:
        examples: list 
            List of examples in the dataset.
        labels: np.array 
            Array of labels in the dataset.
    """
    assert len(grammars) == 2, "Must provide two distinct grammars"
    assert min_seq_len is None or min_seq_len >= 32, "Sequence length must be at least 32"
    assert min_seq_len is None or min_seq_len < max_seq_len, "Minimum sequence length must be less than maximum sequence length"
    # Generate labels
    indicators = ['X', 'Y']
    orders = [[0, 1], [1, 0], [1, 1], [0, 0]]

    examples = [None] * num_examples
    labels = np.zeros(num_examples, dtype=int)
    for _ in range(0, num_examples):
        # Select label
        label = np.random.choice(range(len(orders)))
        order = orders[label]

        # Draw sequence length from beta distribution
        cur_seq_len = max_seq_len if min_seq_len is None else min_seq_len + int(np.random.beta(alpha, beta) * (max_seq_len - min_seq_len))

        if k_split is None:
            split = int(cur_seq_len // 2)
        else:
            # Draw split of sequences from normal distribution centered around middle of sequence
            split = np.clip(int(np.random.normal(cur_seq_len // 2,
                            cur_seq_len * k_split)), 8, cur_seq_len - 8)

        sequence = generate_grammar(
            grammars[order[0]], split) + generate_grammar(grammars[order[1]], cur_seq_len - split)

        # Draw indicator tokens from normal distribution centered around middle of sequence
        if k_indicator is not None:
            idx_one = get_indicator_idx(cur_seq_len, k_indicator)
            idx_two = get_indicator_idx(cur_seq_len, k_indicator)
            if idx_one == idx_two:
                idx_two += np.random.choice([-1, 1])

            sequence[idx_one] = np.random.choice(indicators)
            sequence[idx_two] = np.random.choice(indicators)

            # Compute label
            if sequence[idx_one] == sequence[idx_two]:
                label += len(orders)

        examples[_] = sequence

        labels[_] = label

    return examples, labels


def get_indicator_idx(cur_seq_len, k):
    """
    Select an index for the indicator token based on the current sequence length.
    The index is drawn from a normal distribution centered around the middle of the sequence.

    Args:
        cur_seq_len: int 
            Current sequence length.
        k: float 
            Scale the standard deviation for the indicator tokens.
    
    Returns:
        idx: int 
            Index for the indicator token.
    """
    return np.clip(int(np.random.normal(cur_seq_len // 2, cur_seq_len * k)), 1, cur_seq_len - 2)


def get_split(dataset, *, batch_size=32, validation_split=0.1):
    """
    Split the dataset into training and validation sets.

    Args:
        dataset: Dataset 
            Dataset to be split.
        batch_size: int 
            Batch size for the dataloaders.
        validation_split: float 
            Fraction of the dataset to be used for validation.
    
    Returns:
        train_dataloader: DataLoader 
            Dataloader for the training set.
        val_dataloader: DataLoader 
            Dataloader for the validation set.
    """
    val_size = int(validation_split * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [len(dataset) - val_size, val_size])
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader
