#!/bin/bash
source /w/340/lucbr/miniconda3/bin/activate
export PYTHONPATH="$PYTHONPATH:/u/lucbr/min-gru-transformers/src"
conda activate pytorch_env
python3 experiments/train_model.py --train_dataset_path='exp_64.pt' --model=1