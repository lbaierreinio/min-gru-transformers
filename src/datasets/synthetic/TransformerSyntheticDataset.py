from torch.utils.data import Dataset
import torch


class TransformerSyntheticDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=2048, chunk_size=None):

        if chunk_size:
            for seq in sequences:
                for idx in range(0, len(seq), chunk_size):
                    seq.insert(idx+(chunk_size-1), '[CLS]')
        self.encodings = tokenizer(
            [' '.join(seq) for seq in sequences],
            padding='max_length',
            add_special_tokens=False,
            max_length = max_length,
            return_tensors='pt'
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }
