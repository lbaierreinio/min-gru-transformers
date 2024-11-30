#!/bin/bash
source /w/340/lucbr/miniconda3/bin/activate
export PYTHONPATH="$PYTHONPATH:/u/lucbr/min-gru-transformers/src"
conda activate pytorch_env
python3 experiments/test.py --train_dataset_path='dataset.pt' --validation_dataset_path='dataset.pt' --out_path='out_even_odd.csv' --model=0