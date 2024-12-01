import os
import time

import torch
import torch.profiler

from transformers import AutoTokenizer
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

from models.minGRUSquadQA import MinGRUSquadQA, MinGRUSquadQAConfig 
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
"""
max_lr = 1e-5 #6e-4 * 3
min_lr = max_lr * 0.1
#warmup_steps = int(0.1 * num_training_steps) #5000
epochs = 5 #30
B = 4 #16 # batch size
T = 512 #1024 # sequence length
"""
max_lr = 6e-4 * 3
min_lr = max_lr * 0.1
epochs = 10
B = 10 # batch size
T = 512 # sequence length

# Model configurations
n_layer = 1 #2
hidden_dim = 64 #256
classification_head_dim = 64 #256
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
model.to(device)

#tweaked data loader to include num_examples to work on subset 

NUM_EXAMPLES = 100

train_loader, val_loader = get_squad_v2_dataloaders(tokenizer, batch_size=B, num_examples=NUM_EXAMPLES)
#import pdb; pdb.set_trace()
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
warmup_steps = int(0.1 * num_training_steps)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

for epoch in range(epochs):
    t0 = time.time()
    # Initialize profiler for the epoch
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ] if torch.cuda.is_available() else [torch.profiler.ProfilerActivity.CPU],
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=len(train_loader), repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        # Training mode
        model.train()
        loss_accum = 0.0
        tokens_processed = 0
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            batch_start_time = time.time()
            x, answer_start_idx, answer_end_idx = (
                batch["input_ids"].to(device), # [batch_size, sequence_length]
                batch["answer_start_idx"].to(device).view(-1, 1), # [batch_size, 1]
                batch["answer_end_idx"].to(device).view(-1, 1) # [batch_size, 1]
            )
            y = torch.cat([answer_start_idx, answer_end_idx], dim=-1) # [batch_size, 2]
            optimizer.zero_grad()
            if ampere_gpu:
                # mixed precision training
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, loss = model(x, y)
            else:
                logits, loss = model(x, y)

            loss_accum += loss.detach()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            tokens_in_batch = x.shape[0] * x.shape[1]  
            tokens_processed += tokens_in_batch
            batch_time = time.time() - batch_start_time
            
            if device == "cuda":
                torch.cuda.synchronize()
                batch_memory = torch.cuda.max_memory_allocated()
                torch.cuda.reset_peak_memory_stats()
            else:
                batch_memory = None
            
            # Log per-batch metrics
            print(
                f"Epoch {epoch}, Batch {batch_idx} | Time: {batch_time:.4f}s | "
                f"Tokens: {tokens_in_batch} | "
                f"Memory: {batch_memory} bytes | Loss: {loss.item():.4f}"
            )

            # Step the profiler
            prof.step()
        
        total_flops = sum([e.flops for e in prof.key_averages()])

        # Print/Log training metrics
        if torch.cuda.is_available():
            # wait for all cuda processes to finish to get accurate timing
            torch.cuda.synchronize()
        t1 = time.time()
        avg_loss = loss_accum / len(train_loader)
        cur_lr = scheduler.get_last_lr()[0]
        dt = t1 - t0
        tok_per_sec = tokens_processed / dt
        tokens_processed_in_millions = tokens_processed / 1e6
        flops_in_tera = total_flops / 1e12

        epoch_metrics = f"[Train] Epoch {epoch:4d} | FLOPs (T): {flops_in_tera:.6f} | Tokens (M): {tokens_processed_in_millions:.6f} | loss: {avg_loss:.6f} | cur_lr: {cur_lr:.6f} | dt: {dt:.2f} | tok/sec: {tok_per_sec:.2f}"
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

            epoch_metrics = f"[Val] Epoch {epoch:4d} | val_loss {avg_val_loss:.4f}"
            print(epoch_metrics)
            with open(log_file, "a") as f:
                f.write(f"{epoch_metrics}\n")
        # Export profiler results
    #prof.export_chrome_trace(os.path.join(log_dir, "trace.json"))
    """print(
    prof.key_averages().table(
    row_limit=10,
    )
    )"""