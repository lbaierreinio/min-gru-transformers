#!/bin/bash
source /w/340/lucbr/miniconda3/bin/activate
export PYTHONPATH="$PYTHONPATH:/w/331/lucbr/min-gru-transformers/src"
conda activate pytorch_env
python3 -u experiments/train_model.py --out_path='mingru_4096_memory_test.csv' --dataset_path='mingru_d_4096_1000_hard.pt' --model=0
