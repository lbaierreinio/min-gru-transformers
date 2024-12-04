import os
import time

import torch
from evaluate import load
from transformers import AutoTokenizer
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

from models.MinGRUSquadQA import MinGRUSquadQA, MinGRUSquadQAConfig
from utils.squad_dataloader import get_squad_dataloaders, get_squad_validation_references

# TODO: may want to do graident accumulation
# TODO: may want different accuracies for answerable and non-answerable
# ########################################################
# Set configurations

# Process configurations
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
use_fused = torch.cuda.is_available()
use_compile = True
ampere_gpu = True
if ampere_gpu:
    torch.set_float32_matmul_precision("high") # use tf32 where possible
# autodetect the device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps" # apple silicon
print(f"Using device: {device}")

# Optimizer configurations
max_lr = 6e-4 * 3
min_lr = max_lr * 0.1
warmup_steps = 5000
epochs = 30
B = 64 # batch size

# Model configurations
n_layer = 8
hidden_dim = 768
classification_head_dim = 768
bidirectional=True
#########################################################
# Create model
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
config = MinGRUSquadQAConfig(
    vocab_size=tokenizer.vocab_size,
    n_layer=n_layer,
    hidden_dim=hidden_dim,
    classification_head_dim=classification_head_dim,
    bidirectional=bidirectional
)
model = MinGRUSquadQA(config)

model_size = sum(p.numel() for p in model.parameters())
print(f"Model size: {model_size} parameters")
# NOTE: for reference, the BiDAF baseline from file:///Users/kyungjaelee/school/uoft/f24/csc2516/Project/default-final-project-handout.pdf
#       uses 27,968,705 parameters and achieves F1 ~58, EM ~55 after 25 epochs

model.to(device)

squad_version = "squad" # "squad" for v1, "squad_v2" for v2
if squad_version not in ["squad", "squad_v2"]:
    raise Exception("Invalid SQuAD version provided")
train_loader, val_loader = get_squad_dataloaders(tokenizer, batch_size=B, version=squad_version)
references = get_squad_validation_references(version=squad_version)
squad_metric = load(squad_version)
if use_compile:
    model = torch.compile(model)

# Create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # this clears the existing logs
    f.write("Training hyperparamaters:")
    f.write(f"    {max_lr=}, {min_lr=}, {warmup_steps=}, {epochs=}, batch_size={B}\n")
    f.write(f"Model configurations:")
    f.write(f"    {n_layer=}, {hidden_dim=}, {classification_head_dim=}, {bidirectional=}")
    f.write(f"    {model_size=}")
    f.write(f"Training on task: {squad_version}")

eval_every = 5 # Every n epochs, evaluate EM and F1

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=use_fused)
num_training_steps = epochs * len(train_loader)  # Total number of steps
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)


def forward_batch(batch):
    # Given a batch, format the data and forward through model to get logits and loss
    x, answer_start_idx, answer_end_idx = (
        batch["input_ids"].to(device), # [batch_size, sequence_length]
        batch["answer_start_idx"].to(device).view(-1, 1), # [batch_size, 1]
        batch["answer_end_idx"].to(device).view(-1, 1) # [batch_size, 1]
    )
    y = torch.cat([answer_start_idx, answer_end_idx], dim=-1) # [batch_size, 2]
    mask = (x == tokenizer.pad_token_id)
    if ampere_gpu:
        # mixed precision training
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, loss = model(x, targets=y, mask=mask)
    else:
        logits, loss = model(x, targets=y, mask=mask)

    return logits, loss


def get_predictions(batch, logits):
    """
    Given [B,T,2] tensor of logits representing the scores for each start position
    and end position, return the best overall prediction, while ensuring
    that the selected range is valid.

    Adhere to the format described in https://huggingface.co/spaces/evaluate-metric/squad_v2
    for the outputs.
    """
    ids = batch["id"]
    input_ids = batch["input_ids"]
    token_type_ids = batch["token_type_ids"]
    B, T = input_ids.shape

    start_logits, end_logits = logits[:,:,0], logits[:,:,1] # [B, T]

    # Only consider best top_k start and end positions for efficieny
    top_k = 10
    top_start_scores, top_start_indices = torch.topk(start_logits, k=top_k, dim=1) # [B, top_k]
    top_end_scores, top_end_indices = torch.topk(end_logits, k=top_k, dim=1) # [B, top_k]

    predictions = []
    for b in range(B):
        # Compute all pairwise scores between the top k best start/end positions and take the best
        # Use "no answer" prediction as default value for v2
        max_score = (start_logits[b, 0] + end_logits[b, 0]) if squad_version == "squad_v2" else float("-inf")
        best_range = (0, 0)
        for i in range(top_k):
            for j in range(top_k):
                start_idx = top_start_indices[b, i].item()
                end_idx = top_end_indices[b, j].item()
                # Do not consider predictions with invalid start/end position combinations
                if end_idx <= start_idx:
                    continue
                # Do not consider predictions that lie outside context
                # NOTE: token_type_ids will be a list of 0s (tokens corresponding to question), followed by
                #       a list of 1s (tokens corresponding to context), again followed by a list of 0s (padding)
                context_start_idx = 0
                while token_type_ids[b][context_start_idx] == 0:
                    context_start_idx += 1
                context_end_idx = T - 1
                while token_type_ids[b][context_end_idx] == 0:
                    context_end_idx -= 1
                if not(
                    (context_start_idx <= start_idx <= context_end_idx) and
                    (context_start_idx <= end_idx <= context_end_idx)
                ):
                    continue

                # 
                score = top_start_scores[b, i] + top_end_scores[b, j]
                if score > max_score:
                    max_score = score
                    best_range = (start_idx, end_idx)
        if best_range == (0,0):
            prediction = {"id": ids[b], "prediction_text": ""}
        else:
            prediction_text = tokenizer.decode(input_ids[b][best_range[0]:best_range[1]+1])
            prediction = {"id": ids[b], "prediction_text": prediction_text}
        
        if squad_version == "squad_v2":
            prediction["no_answer_probability"] = 1.0 if best_range == (0,0) else 0.
        predictions.append(prediction)

    return predictions


for i in range(epochs):
    t0 = time.time()
    # optimize
    model.train()
    loss_accum = 0.0
    tokens_processed = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        logits, loss = forward_batch(batch)
        loss_accum += loss.detach()
        loss.backward()
        optimizer.step()
        scheduler.step()
        tokens_processed += batch["input_ids"].shape[0] * batch["input_ids"].shape[1] # batch_size * sequence_length
        
    # Print/Log training metrics
    if torch.cuda.is_available():
        # wait for all cuda processes to finish to get accurate timing
        torch.cuda.synchronize()
    t1 = time.time()
    avg_train_loss = loss_accum / len(train_loader)
    cur_lr = scheduler.get_last_lr()[0]
    dt = t1 - t0
    tok_per_sec = tokens_processed / dt

    # eval
    model.eval()
    with torch.no_grad():
        should_get_predictions = i % eval_every == 0
        val_loss_accum = 0.0

        # As we're iterating through the batch, get predictions in the format {"id": ..., "prediction": ...}
        predictions = []
        for batch in tqdm(val_loader):
            logits, loss = forward_batch(batch)
            val_loss_accum += loss.detach()
            # Only perform eval every few epochs
            if should_get_predictions:
                predictions.extend(get_predictions(batch, logits))
        
        avg_val_loss = val_loss_accum / len(val_loader)
        if should_get_predictions:
            results = squad_metric.compute(predictions=predictions, references=references)
            em_key = "exact" if squad_version == "squad_v2" else "exact_match"
            em, f1 = results[em_key], results["f1"]
            em_str = f"EM: {em:.4f}"
            f1_str = f"F1: {f1:.4f}"
        else:
            em_str = f"EM: N/A"
            f1_str = f"F1: N/A"
        epoch_metrics = f"Epoch {i:4d} | train_loss: {avg_train_loss:.6f} | val_loss: {avg_val_loss:.6f} | cur_lr: {cur_lr:.6f} | dt (train): {dt:.2f} | tok/sec (train): {tok_per_sec:.2f} | {em_str} | {f1_str}"
        print(epoch_metrics)
        with open(log_file, "a") as f:
            f.write(f"{epoch_metrics}\n")

# Save last model
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}
torch.save(checkpoint, 'checkpoint.pth')
print("Checkpoint saved!")
