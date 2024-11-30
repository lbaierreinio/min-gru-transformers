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
Long-Context Task for both Summarization & Long Range Dependency. Specifically,
the model is asked to do two things: (a) summarize the order in which the subsequences
appear and (b) determine if the two indicator tokens are the same.

'''


def generate_dataset8(*, seq_len, num_examples, grammars, num_labels, num_subsequences, start, end, orders, even, seed=42):

    assert seq_len % num_subsequences == 0, "Sequence length must be divisible by number of subsequences"
    assert start < end, "Start must be less than end"
    assert start >= 0, "Start must be greater than or equal to 0"
    assert end <= seq_len, "End must be less than or equal to sequence length"
    assert num_labels >= 2, "Number of labels must be greater than or equal to 2"
    assert len(grammars) >= 1, "Must specify at least one grammar"
    assert len(
        orders) == num_labels // 2, "Number of orders must be equal to half the number of labels"

    s, e = int(start // 2), int(end // 2)
    # Generate labels
    indicators = ['X', 'Y']
    halved_num_labels = int(num_labels // 2)

    examples = np.zeros((num_examples, seq_len), dtype=object)
    labels = np.zeros(num_examples, dtype=int)
    for _ in range(0, num_examples):
        label = np.random.choice(range(halved_num_labels))
        order = orders[label]
        subseq_len = int(seq_len / num_subsequences)
        sequence = []
        for i in range(0, num_subsequences):
            sequence += generate_grammar(grammars[order[i]], subseq_len)

        idx_one = np.random.randint(s, e)
        idx_two = idx_one
        while idx_one == idx_two:
            idx_two = np.random.randint(s, e)

        idx_one = idx_one * 2
        idx_two = idx_two * 2

        if not even:
            idx_one = max(1, idx_one - 1)
            idx_two = max(1, idx_two - 1)
            if idx_one == idx_two:
                idx_two += 2

        sequence[idx_one] = np.random.choice(indicators)
        sequence[idx_two] = np.random.choice(indicators)

        # Compute ground truth
        first = sequence[idx_one]
        second = sequence[idx_two]
        if first == second:
            label += halved_num_labels

        examples[_] = sequence

        labels[_] = label

    return examples, labels


def get_split(dataset, batch_size=32, validation_split=0.1, test_split=0.1, seed=42):
    val_size = int(validation_split * len(dataset))
    test_size = int(test_split * len(dataset))
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [len(dataset) - val_size - test_size, val_size, test_size])
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader
