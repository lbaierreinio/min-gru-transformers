import torch
import numpy as np
from torch.utils.data import DataLoader

'''
Given a set of rules for a grammar (in the form of a dict),
and a max sequence length, generate a sequence of tokens
from said grammar.
'''


def generate_grammar(rules, seq_len):
    sequence = []
    state = 'S'

    while len(sequence) < seq_len:
        current_rules = rules[state]
        next_state_idx = np.random.choice(
            np.arange(len(current_rules)), p=list(map(lambda x: x[0], current_rules)))
        state = current_rules[next_state_idx][1]
        sequence.append(state)

    return sequence


'''
Long-Context Task for both Summarization & Long Range Dependencies. Specifically,
the model is asked to solve two tasks simultaneously: 
(a) Recall which order the two grammars appeared in.
(b) Determine if the two indicator tokens are the same.
'''


def generate_dataset8(*, seq_len, num_examples, grammars, alpha, beta, k_split, k_indicator, seed=42):
    assert len(grammars) == 2, "Must provide two distinct grammars"
    assert seq_len >= 32, "Sequence length must be at least 32"
    assert num_examples > 100, "Number of examples must be greater than 100"
    # Generate labels
    indicators = ['X', 'Y']
    orders = [[0, 1], [1, 0], [1, 1], [0, 0]]

    examples = np.zeros((num_examples, seq_len), dtype=object)
    labels = np.zeros(num_examples, dtype=int)
    for _ in range(0, num_examples):
        # Select label
        label = np.random.choice(range(len(orders)))
        order = orders[label]

        # Draw sequence length from beta distribution
        cur_seq_len = min(32, int(np.random.beta(alpha, beta) * seq_len))

        # Draw split of sequences from normal distribution centered around middle of sequence
        split = np.clip(int(np.random.normal(cur_seq_len // 2,
                        cur_seq_len * 0.05)), 8, cur_seq_len - 8)

        sequence = generate_grammar(
            grammars[order[0]], split) + generate_grammar(grammars[order[1]], cur_seq_len - split)

        # Draw indicator tokens from normal distribution centered around middle of sequence
        idx_one = get_indicator_idx(cur_seq_len, k_indicator)
        idx_two = get_indicator_idx(cur_seq_len, k_indicator)
        if idx_one == idx_two:
            idx_two += np.random.choice([-1, 1])

        sequence[idx_one] = np.random.choice(indicators)
        sequence[idx_two] = np.random.choice(indicators)

        # Compute label
        if sequence[idx_one] == sequence[idx_two]:
            label += len(orders)

        # Pad sequence
        sequence = sequence + ['PAD'] * (seq_len - len(sequence))

        examples[_] = sequence

        labels[_] = label

    return examples, labels


def get_indicator_idx(cur_seq_len, k):
    return np.clip(int(np.random.normal(cur_seq_len // 2, cur_seq_len * k)), 1, cur_seq_len - 2)


def get_split(dataset, *, batch_size=32, validation_split=0.1, seed=42):
    val_size = int(validation_split * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [len(dataset) - val_size, val_size])
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader