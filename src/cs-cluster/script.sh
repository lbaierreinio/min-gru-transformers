#!/bin/bash
source /w/340/lucbr/miniconda3/bin/activate
export PYTHONPATH="$PYTHONPATH:/w/331/lucbr/min-gru-transformers/src"
conda activate pytorch_env
python3 experiments/train_mingru.py --dataset_path='mingru_d_indicator_512_12000.pt' --out_path='mingru_indicator_512_12000.csv'
