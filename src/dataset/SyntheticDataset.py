from torch.utils.data import Dataset
import torch
class SyntheticDataset(Dataset):
  def __init__(self, sequences, labels, tokenizer, max_length):
      self.encodings = tokenizer(
          [' '.join(seq) for seq in sequences],
          truncation=False,
          padding=True,
          add_special_tokens=False,
          max_length=max_length,
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