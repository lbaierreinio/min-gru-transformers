import torch
from transformers import AutoTokenizer
from models.MinGRUSquadQA import MinGRUSquadQA, MinGRUSquadQAConfig
from utils.squad_dataloader import get_squad_dataloaders, get_squad_validation_references
from datasets import load_dataset
import random

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# load model configuration to evaluate
n_layer = 10
hidden_dim = 302
classification_head_dim = hidden_dim
bidirectional = True

config = MinGRUSquadQAConfig(
    vocab_size=tokenizer.vocab_size,
    n_layer=n_layer,
    hidden_dim=hidden_dim,
    classification_head_dim=classification_head_dim,
    bidirectional=bidirectional
)

model = MinGRUSquadQA(config)

# load the checkpoint
checkpoint = torch.load("/w/246/indiravats/log/checkpoints/10-302-T-best-combined.pth", weights_only=True)
state_dict = checkpoint["model_state_dict"]
new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# load validation data
batch_size = 32
_, val_loader = get_squad_dataloaders(tokenizer, batch_size=batch_size, version="squad")

# create new reference including 'question' and 'title'
def get_extended_validation_references(version="squad"):
    # Load the dataset
    val_dataset = load_dataset(f"rajpurkar/{version}", split="validation")
    
    # Keep only id, answers, question, and title
    extended_references = []
    for example in val_dataset:
        extended_references.append({
            "id": example["id"],
            "answers": example["answers"],
            "question": example["question"],
            "title": example["title"],
        })
    return extended_references

# generate the new validation references
val_ref = get_extended_validation_references(version="squad")

val_data_dict = {ex["id"]: ex for ex in val_ref}

def get_question_type(question):
    if question.lower().startswith(("who", "what", "when", "where", "why", "how")):
        return question.split()[0].lower()
    return "other"

correct_examples = []
incorrect_examples = []
qtype_correct_counts = {}
qtype_total_counts = {}
title_correct_counts = {}
title_total_counts = {}

for batch in val_loader:
    with torch.no_grad():
        input_ids = batch["input_ids"].to(device)
        mask = (input_ids == tokenizer.pad_token_id).to(device)

        logits, _ = model(input_ids, mask=mask)
        start_logits, end_logits = logits[:, :, 0], logits[:, :, 1]

        start_pred = start_logits.argmax(dim=1)
        end_pred = end_logits.argmax(dim=1)

        for b in range(len(batch["id"])):
            ex_id = batch["id"][b]
            input_ids_list = batch["input_ids"][b].tolist()

            pred_span = input_ids_list[start_pred[b].item(): end_pred[b].item() + 1]
            predicted_answer = tokenizer.decode(pred_span, skip_special_tokens=True).strip()

            corr_start = batch["answer_start_idx"][b].item()
            corr_end = batch["answer_end_idx"][b].item()
            corr_span = input_ids_list[corr_start: corr_end + 1]
            correct_answer = tokenizer.decode(corr_span, skip_special_tokens=True).strip()

            original_example = val_data_dict[ex_id]
            original_question = original_example["question"]
            original_title = original_example["title"]

            is_correct = (predicted_answer == correct_answer)

            # Store examples
            if is_correct:
                correct_examples.append((original_question, original_title, predicted_answer, correct_answer))
            else:
                incorrect_examples.append((original_question, original_title, predicted_answer, correct_answer))

            # update question type counts
            qtype = get_question_type(original_question)
            qtype_total_counts[qtype] = qtype_total_counts.get(qtype, 0) + 1
            if is_correct:
                qtype_correct_counts[qtype] = qtype_correct_counts.get(qtype, 0) + 1

            # update title counts
            title_total_counts[original_title] = title_total_counts.get(original_title, 0) + 1
            if is_correct:
                title_correct_counts[original_title] = title_correct_counts.get(original_title, 0) + 1

# print examples
random.seed(42)

random_correct_examples = random.sample(correct_examples, min(10, len(correct_examples)))
print("===== 10 CORRECT EXAMPLES =====")
for example in random_correct_examples:
    original_question, original_title, pred_ans, corr_ans = example
    print("Title:", original_title)
    print("Question:", original_question)
    print("Predicted Answer:", pred_ans)
    print("Correct Answer:  ", corr_ans)
    print("------------------------------------------------------")

random_incorrect_examples = random.sample(incorrect_examples, min(10, len(incorrect_examples)))
print("===== 10 INCORRECT EXAMPLES =====")
for example in random_incorrect_examples:
    original_question, original_title, pred_ans, corr_ans = example
    print("Title:", original_title)
    print("Question:", original_question)
    print("Predicted Answer:", pred_ans)
    print("Correct Answer:  ", corr_ans)
    print("------------------------------------------------------")

# print question type analysis
print("===== QUESTION TYPE ANALYSIS =====")
for qtype in sorted(qtype_total_counts.keys()):
    total = qtype_total_counts[qtype]
    correct = qtype_correct_counts.get(qtype, 0)
    accuracy = correct / total if total > 0 else 0
    print(f"Question Type: {qtype:>5} | Total: {total:>5} | Correct: {correct:>5} | Accuracy: {accuracy:.2f}")

# print title analysis
print("===== TITLE ANALYSIS =====")
for title in sorted(title_total_counts.keys(), key=lambda t: title_total_counts[t], reverse=True):
    total = title_total_counts[title]
    correct = title_correct_counts.get(title, 0)
    accuracy = correct / total if total > 0 else 0
    print(f"Title: {title:>20} | Total: {total:>5} | Correct: {correct:>5} | Accuracy: {accuracy:.2f}")
