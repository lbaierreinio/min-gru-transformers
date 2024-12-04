import os
import time

import torch
from evaluate import load
from transformers import AutoTokenizer
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

import torch.profiler

from models.MinGRUSquadQA import MinGRUSquadQA, MinGRUSquadQAConfig
from utils.squad_dataloader import get_squad_v2_dataloaders, get_squad_v2_validation_references

# TODO: may want to do graident accumulation
# TODO: explore GRU bidirectionality
# TODO: may want to checkpoint model at the end (/ between epochs)
# TODO: may want different accuracies for answerable and non-answerable
# ########################################################
# Set configurations

# Process configurations
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
use_fused = torch.cuda.is_available()
use_compile = False
ampere_gpu = False
if ampere_gpu:
    torch.set_float32_matmul_precision("high") # use tf32 where possible
# autodetect the device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps" # apple silicon
print(f"Using device: {device}")

"""
# Optimizer configurations
max_lr = 6e-4 * 3
min_lr = max_lr * 0.1
warmup_steps = 5000
epochs = 30
B = 16 # batch size

"""
# Testing configs
max_lr = 1e-3 
min_lr = max_lr * 0.1
epochs = 10
B = 10 

# Model configurations
n_layer = 2
hidden_dim = 256
classification_head_dim = 256
#########################################################
# Create model
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
config = MinGRUSquadQAConfig(
    vocab_size=tokenizer.vocab_size,
    n_layer=n_layer,
    hidden_dim=hidden_dim,
    classification_head_dim=classification_head_dim
)
model = MinGRUSquadQA(config)

model_size = sum(p.numel() for p in model.parameters())
print(f"Model size: {model_size} parameters")
# NOTE: for reference, the BiDAF baseline from file:///Users/kyungjaelee/school/uoft/f24/csc2516/Project/default-final-project-handout.pdf
#       uses 27,968,705 parameters and achieves F1 ~58, EM ~55 after 25 epochs

model.to(device)

# num examples could be set here if needed for debugging 
NUM_EXAMPLES = 1000
train_loader, val_loader = get_squad_v2_dataloaders(tokenizer, batch_size=B, num_examples=NUM_EXAMPLES)
references = get_squad_v2_validation_references(num_examples=NUM_EXAMPLES)
"""
train_loader, val_loader = get_squad_v2_dataloaders(tokenizer, batch_size=B)
references = get_squad_v2_validation_references()
"""
squad_metric = load("squad_v2")
if use_compile:
    model = torch.compile(model)

# Create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass
eval_every = 5 # Every n epochs, evaluate EM and F1

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=use_fused)
num_training_steps = epochs * len(train_loader)  # Total number of steps

#remove if set before
warmup_steps = int(0.1 * num_training_steps)

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
    top_k = 20
    top_start_scores, top_start_indices = torch.topk(start_logits, k=top_k, dim=1) # [B, top_k]
    top_end_scores, top_end_indices = torch.topk(end_logits, k=top_k, dim=1) # [B, top_k]

    predictions = []
    for b in range(B):
        # Compute all pairwise scores between the top k best start/end positions and take the best
        # Use "no answer" prediction as default value
        max_score = start_logits[b, 0] + end_logits[b, 0]
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
                score = top_start_scores[b, i]= + top_end_scores[b, j]
                if score > max_score:
                    max_score = score
                    best_range = (start_idx, end_idx)
        if best_range == (0,0):
            prediction = {"id": ids[b], "prediction_text": "", "no_answer_probability": 1.0}
        else:
            prediction_text = tokenizer.decode(input_ids[b][best_range[0]:best_range[1]+1])
            prediction = {"id": ids[b], "prediction_text": prediction_text, "no_answer_probability": 0.}
        predictions.append(prediction)

    return predictions


for i in range(epochs):
    t0 = time.time()

    # resets memory to measure usage for that particular epoch
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)

    # optimize
    loss_accum = 0.0
    tokens_processed = 0
    epoch_flops = 0

    model.train()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ] if torch.cuda.is_available() else [torch.profiler.ProfilerActivity.CPU],
        record_shapes=False,
        profile_memory=True,
        with_flops=True,
    ) as prof:
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            logits, loss = forward_batch(batch)
            loss_accum += loss.detach()
            loss.backward()
            optimizer.step()
            scheduler.step()
            tokens_processed += batch["input_ids"].shape[0] * batch["input_ids"].shape[1] # batch_size * sequence_length

    epoch_flops += sum(
    [event.flops for event in prof.key_averages() if event.flops is not None]) # returns flops for current epoch
        
    # Print/Log training metrics
    if torch.cuda.is_available():
        # wait for all cuda processes to finish to get accurate timing
        torch.cuda.synchronize()
    t1 = time.time()
    avg_loss = loss_accum / len(train_loader)
    cur_lr = scheduler.get_last_lr()[0]
    dt = t1 - t0
    tok_per_sec = tokens_processed / dt
    flops_in_tera = epoch_flops / 1e12

    if device == "cuda":
        max_memory = torch.cuda.max_memory_allocated(device=device) / (1024 * 1024)  # convert to MB
    else:
        max_memory = max(
        event.cpu_memory_usage
        for event in prof.key_averages()
        if event.cpu_memory_usage is not None)
        max_memory = max_memory / (1024 * 1024)

    epoch_metrics = (
        f"[Train] Epoch {i:4d} | loss: {avg_loss:.6f} | cur_lr: {cur_lr:.6f} "
        f"| dt: {dt:.2f} | tok/sec: {tok_per_sec:.2f} | Memory (MB): {max_memory:.6f} | FLOPs (T): {flops_in_tera:.6f}"
    )
    print(epoch_metrics)
    with open(log_file, "a") as f:
        f.write(f"{epoch_metrics}\n")

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
            # Only perform eval every 5 epochs
            if should_get_predictions:
                predictions.extend(get_predictions(batch, logits))
        
        avg_val_loss = val_loss_accum / len(val_loader)
        if should_get_predictions:
            results = squad_metric.compute(predictions=predictions, references=references)
            em, f1 = results["exact"], results["f1"]
            epoch_metrics = f"[Val] Epoch {i:4d} | val_loss: {avg_val_loss:.4f} | EM: {em:.4f} | F1: {f1:.4f}"
        else:
            epoch_metrics = f"[Val] Epoch {i:4d} | val_loss: {avg_val_loss:.4f} | EM: N/A | F1: N/A"
            
        print(epoch_metrics)
        with open(log_file, "a") as f:
            f.write(f"{epoch_metrics}\n")
