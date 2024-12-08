import os
import torch
import time
import argparse
import torch.profiler
from train.utility import train, profile
from dataclasses import dataclass
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets.synthetic.utility import get_split
from models.MinGRUSynthetic import MinGRUSynthetic

@dataclass
class MinGRUConfig:
    """
    Configuration for minGRU model.
    """
    name: str = 'mingru'
    num_layers: int = 3
    embedding_dim: int = 384
    bidirectional: bool = True

def main():
    # Define model & dataset
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        help='Validation dataset path')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_path = args.dataset_path

    if not os.path.exists(dataset_path):
        raise ValueError("Paths must point to a valid file")

    dataset = torch.load(dataset_path)
    train_dataloader, val_dataloader = get_split(dataset, batch_size=16)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    config = MinGRUConfig()
    model = MinGRUSynthetic(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=config.embedding_dim,
        num_layers=config.num_layers,
        bidirectional=config.bidirectional,
        num_classes=4
    ).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # Train model
    train(model, train_dataloader, val_dataloader, 20, loss_fn, optimizer, early_stopping_threshold=0.95)

    # Get a batch from validation set
    batch = next(iter(val_dataloader))
    input = batch['input_ids'].to(device).to(device)
    attention_mask = ~batch['attention_mask'].bool().to(device)

    model.eval()

    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)            

    # Obtain results
    p_max_memory, p_time_elapsed, p_output = profile(model, input, attention_mask, device, is_sequential=False)
    s_max_memory, s_time_elapsed, s_output = profile(model, input, attention_mask, device, is_sequential=True)
    s_pred = torch.argmax(s_output, dim=1)
    p_pred = torch.argmax(p_output, dim=1)
    
    # Assert that predictions & logits are the same
    assert torch.allclose(s_output, p_output, rtol=1e-4, atol=1e-6)
    assert torch.equal(s_pred, p_pred)

    print(f"GPU: {gpu_name}")
    print(f"Parallel max memory: {p_max_memory} MB")
    print(f"Sequential max memory: {s_max_memory} MB")
    print(f"Parallel time: {p_time_elapsed} s")
    print(f"Sequential time: {s_time_elapsed} s")


if __name__ == '__main__':
    main()
