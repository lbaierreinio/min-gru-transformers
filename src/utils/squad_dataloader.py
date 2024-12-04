from functools import partial

from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import DataCollatorWithPadding


def preprocess_function(tokenizer, examples):
    """Given examples (a dict containing questions, contexts, and answers) perform minimal cleanup
    of the questions and contexts (in our case, just strip leading/trailing whitespace), tokenize
    their concatenation using the provided tokenizer, and finally add the start and end positions
    of the answer in the tokenized sequence.

    Each tokenized sequence will have the form: [CLS] question [SEP] context [SEP]
    """
    questions = [q.strip() for q in examples["question"]]
    contexts = [c.strip() for c in examples["context"]]
    inputs = tokenizer(
        questions,
        contexts,
        truncation=False, # no limit on max length
        return_offsets_mapping=True,
        padding=False, # allow arbitrary sequence lengths (in-batch padding handled in DataCollator)
    )
    
    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]

    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]

        # unanswerable question
        if not answer["answer_start"]:
            start_positions.append(0)
            end_positions.append(0)
            continue

        # Answerable question
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])

        # Compute token index of answer start and end
        sequence_ids = inputs.sequence_ids(i) # NOTE: this is different from inputs["token_type_ids"] in how it handles special tokens
        context_start = sequence_ids.index(1) # question tokens have type_id=0, context tokens have type_id=1 
        context_end = len(sequence_ids) - 1
        while sequence_ids[context_end] != 1:
            context_end -= 1

        idx = context_start
        while offset[idx][0] < start_char:
            idx += 1
        start_positions.append(idx)

        idx = context_end
        while offset[idx][1] > end_char:
            idx -= 1
        end_positions.append(idx)

    inputs["answer_start_idx"] = start_positions
    inputs["answer_end_idx"] = end_positions
    inputs.pop("attention_mask") # remove
    return inputs


def get_squad_v2_dataloaders(tokenizer, batch_size, num_examples=None):
    # Load datasets from HuggingFace
    train_dataset = load_dataset("rajpurkar/squad_v2", split="train")
    val_dataset = load_dataset("rajpurkar/squad_v2", split="validation")

    # selects a subset of examples, if arg is specified 
    if num_examples is not None:
        train_dataset = train_dataset.select(range(int(0.8*num_examples)))
        val_dataset = val_dataset.select(range(int(0.2*num_examples)))

    # Preprocess and tokenize
    cols_to_remove = set(train_dataset.column_names)
    cols_to_remove.remove("id") # keep id to do eval
    train_dataset = train_dataset.map(
        partial(preprocess_function, tokenizer),
        batched=True,
        remove_columns=cols_to_remove,
    )
    val_dataset = val_dataset.map(
        partial(preprocess_function, tokenizer),
        batched=True,
        remove_columns=cols_to_remove,
    )

    padding_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # Custom collate function to handle 'id'
    def collate_fn(batch):
        # Extract ids
        ids = [example.pop("id") for example in batch]
        # Collate everything else using DataCollatorWithPadding
        tokenized_batch = padding_collator(batch)
        tokenized_batch["id"] = ids  # Add ids back to the batch
        return tokenized_batch

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    return train_dataloader, val_dataloader

def get_squad_v2_validation_references(num_examples=None):
    val_dataset = load_dataset("rajpurkar/squad_v2", split="validation")

    if num_examples is not None:
      val_dataset = val_dataset.select(range(int(0.2*num_examples)))

    # Remove everything except id and answers
    cols_to_remove = val_dataset.column_names
    cols_to_remove.remove("id")
    cols_to_remove.remove("answers")
    val_dataset = val_dataset.map(remove_columns=cols_to_remove)
    return list(val_dataset)
