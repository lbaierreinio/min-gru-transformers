import os
import time

import torch
from transformers import AutoTokenizer
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

from models.minGRUSquadQA import minGRUSquadQA, minGRUSquadQAConfig
from utils.squad_dataloader import get_squad_v2_dataloaders

# TODO: may want to dograident accumulation
# TODO: explore GRU bidirectionality
# TODO: may want to checkpoint model at the end (/ between epochs)
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

# Optimizer configurations
max_lr = 6e-4 * 3
min_lr = max_lr * 0.1
warmup_steps = 5000
epochs = 30
B = 16 # batch size
T = 1024 # sequence length

# Model configurations
n_layer = 2
hidden_dim = 256
classification_head_dim = 256
#########################################################
# Create model
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
config = minGRUSquadQAConfig(
    vocab_size=tokenizer.vocab_size,
    n_layer=n_layer,
    hidden_dim=hidden_dim,
    classification_head_dim=classification_head_dim
)
model = minGRUSquadQA(config)
model.to(device)

train_loader, val_loader = get_squad_v2_dataloaders(tokenizer, batch_size=B)
import pdb; pdb.set_trace()
if use_compile:
    model = torch.compile(model)

# Create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=use_fused)
num_training_steps = epochs * len(train_loader)  # Total number of steps
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

for i in range(epochs):
    t0 = time.time()
    # optimize
    model.train()
    loss_accum = 0.0
    tokens_processed = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        x, answer_start_idx, answer_end_idx = (
            batch["input_ids"].to(device), # [batch_size, sequence_length]
            batch["answer_start_idx"].to(device).view(-1, 1), # [batch_size, 1]
            batch["answer_end_idx"].to(device).view(-1, 1) # [batch_size, 1]
        )
        y = torch.cat([answer_start_idx, answer_end_idx], dim=-1) # [batch_size, 2]
        if ampere_gpu:
            # mixed precision training
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, loss = model(x, y)
        else:
            logits, loss = model(x, y)
        loss_accum += loss.detach()
        loss.backward()
        optimizer.step()
        scheduler.step()

        tokens_processed += x.shape[0] * x.shape[1] # batch_size * sequence_length
        
    # Print/Log training metrics
    if torch.cuda.is_available():
        # wait for all cuda processes to finish to get accurate timing
        torch.cuda.synchronize()
    t1 = time.time()
    avg_loss = loss_accum / len(train_loader)
    cur_lr = scheduler.get_last_lr()[0]
    dt = t1 - t0
    tok_per_sec = tokens_processed / dt
    epoch_metrics = f"[Train] Epoch {i:4d} | loss: {avg_loss:.6f} | cur_lr: {cur_lr:.6f} | dt: {dt:.2f} | tok/sec: {tok_per_sec:.2f}"
    print(epoch_metrics)
    with open(log_file, "a") as f:
        f.write(f"{epoch_metrics}\n")

    # eval
    model.eval()
    with torch.no_grad():
        val_loss_accum = 0.0
        for batch in tqdm(val_loader):
            x, answer_start_idx, answer_end_idx = (
                batch["input_ids"].to(device), # [batch_size, sequence_length]
                batch["answer_start_idx"].to(device).view(-1, 1), # [batch_size, 1]
                batch["answer_end_idx"].to(device).view(-1, 1) # [batch_size, 1]
            )
            y = torch.cat([answer_start_idx, answer_end_idx], dim=-1) # [batch_size, 2]
            if ampere_gpu:
                # mixed precision training
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, loss = model(x, y)
            else:
                logits, loss = model(x, y)
            val_loss_accum += loss.detach()
        
        avg_val_loss = val_loss_accum / len(val_loader)
        epoch_metrics = f"[Val] Epoch {i:4d} | val_loss {val_loss_accum.item():.4f}"
        with open(log_file, "a") as f:
            f.write(f"{epoch_metrics}\n")
