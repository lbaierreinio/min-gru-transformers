import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from transformers import AutoTokenizer
from load_locov1 import gather_loco_training_examples
from lm.minGRULM import MinGRULM
from sentence_transformers import datasets

# initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
tokenizer.model_max_length = 20000 # default is 512
# build the vocabulary size for the model
vocab_size = tokenizer.vocab_size
loco_example_count = 10
loco_evaluation_set_count = 3
threshold_for_negatives = -1  # not sure what to put here
negatives_per_query = 5  # not sure what to put here
loss_choice = "contrastive_loss"  # not sure what to put here
use_negatives_from_same_dataset_for_multidataset_finetuning = True
long_context_training_examples, long_context_validation_examples = gather_loco_training_examples(
    loco_example_count,
    loco_evaluation_set_count,
    threshold_for_negatives,
    negatives_per_query,
    loss_choice,
    use_negatives_from_same_dataset_for_multidataset_finetuning
)
num_examples_train = 10
num_examples_test = 3

#the loco_example_count was not actually reducing example size
long_context_training_examples = long_context_training_examples[:num_examples_train]
long_context_validation_examples = long_context_validation_examples[:num_examples_test]
def tokenize_input_examples(input_examples, tokenizer):
    tokenized_inputs = []
    for example in input_examples:
        # Tokenize the query and passage separately
        query_inputs = tokenizer(
            example.texts[0],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        passage_inputs = tokenizer(
            example.texts[1],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        # Combine the tokenized inputs into a single dictionary
        inputs = {
            'input_ids_query': query_inputs['input_ids'].squeeze(0),
            'attention_mask_query': query_inputs['attention_mask'].squeeze(0),
            'input_ids_passage': passage_inputs['input_ids'].squeeze(0),
            'attention_mask_passage': passage_inputs['attention_mask'].squeeze(0),
        }
        # Append the inputs and label
        tokenized_inputs.append((inputs, example.label))
    return tokenized_inputs

# Tokenize the training and validation examples
tokenized_train_examples = tokenize_input_examples(long_context_training_examples, tokenizer)
tokenized_val_examples = tokenize_input_examples(long_context_validation_examples, tokenizer)

class ContrastiveDataset(Dataset):
    def __init__(self, tokenized_examples):
        self.examples = tokenized_examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        inputs, label = self.examples[idx]
        return inputs, label

def collate_fn(batch):
    batch_inputs = {
        'input_ids_query': [],
        'attention_mask_query': [],
        'input_ids_passage': [],
        'attention_mask_passage': []
    }
    labels = []
    for inputs, label in batch:
        batch_inputs['input_ids_query'].append(inputs['input_ids_query'])
        batch_inputs['attention_mask_query'].append(inputs['attention_mask_query'])
        batch_inputs['input_ids_passage'].append(inputs['input_ids_passage'])
        batch_inputs['attention_mask_passage'].append(inputs['attention_mask_passage'])
        labels.append(label)
    batch_inputs['input_ids_query'] = torch.stack(batch_inputs['input_ids_query'])
    batch_inputs['attention_mask_query'] = torch.stack(batch_inputs['attention_mask_query'])
    batch_inputs['input_ids_passage'] = torch.stack(batch_inputs['input_ids_passage'])
    batch_inputs['attention_mask_passage'] = torch.stack(batch_inputs['attention_mask_passage'])
    labels = torch.tensor(labels)
    return batch_inputs, labels

train_dataset = ContrastiveDataset(tokenized_train_examples)
val_dataset = ContrastiveDataset(tokenized_val_examples)

train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

model = MinGRULM(
    num_tokens=vocab_size,
    input_dim=768,  # to match BERT-base-uncased embedding size
    hidden_dim=768,
    num_layers=2,
)

import torch

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CosineEmbeddingLoss()  # as its contrastive loss? not sure if i need to write my own fxn for contrastive loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_epochs = 100

def masked_mean(outputs, attention_mask):
  """mean of the output embeddings across the sequence length dimension, taking into account the attention mask to ignore padding tokens"""

  attention_mask = attention_mask.unsqueeze(-1).expand(outputs.size()).float()
  outputs = outputs * attention_mask
  summed = torch.sum(outputs, dim=1)
  counts = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
  mean = summed / counts
  return mean  # shape [batch_size, hidden_dim]

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_inputs, labels in train_dataloader:
        optimizer.zero_grad()

        input_ids_query = batch_inputs['input_ids_query'].to(device)
        attention_mask_query = batch_inputs['attention_mask_query'].to(device)
        input_ids_passage = batch_inputs['input_ids_passage'].to(device)
        attention_mask_passage = batch_inputs['attention_mask_passage'].to(device)
        labels = labels.to(device).float()

        query_outputs, _ = model(input_ids_query)
        passage_outputs, _ = model(input_ids_passage)

        query_embeddings = masked_mean(query_outputs, attention_mask_query)
        passage_embeddings = masked_mean(passage_outputs, attention_mask_passage)

        # adjust labels for CosineEmbeddingLoss
        targets = labels * 2 - 1  # labels: 1 or 0 -> targets: 1 or -1

        loss = criterion(query_embeddings, passage_embeddings, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)

    # val loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_inputs, labels in val_dataloader:
            input_ids_query = batch_inputs['input_ids_query'].to(device)
            attention_mask_query = batch_inputs['attention_mask_query'].to(device)
            input_ids_passage = batch_inputs['input_ids_passage'].to(device)
            attention_mask_passage = batch_inputs['attention_mask_passage'].to(device)
            labels = labels.to(device).float()

            # get embeddings
            query_outputs, _ = model(input_ids_query)
            passage_outputs, _ = model(input_ids_passage)

            # compute masked mean
            query_embeddings = masked_mean(query_outputs, attention_mask_query)
            passage_embeddings = masked_mean(passage_outputs, attention_mask_passage)

            # adjusting labels for CosineEmbeddingLoss
            targets = labels * 2 - 1  # labels: 1 or 0 -> targets: 1 or -1

            loss = criterion(query_embeddings, passage_embeddings, targets)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_dataloader)

    print(f"Epoch: {epoch} | Train loss: {avg_loss:.3f} | Val loss: {avg_val_loss:.3f}")
