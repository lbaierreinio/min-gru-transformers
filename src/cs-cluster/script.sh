#!/bin/bash
source /w/340/lucbr/miniconda3/bin/activate
export PYTHONPATH="$PYTHONPATH:/w/331/lucbr/min-gru-transformers/src"
conda activate pytorch_env
python3 -u experiments/train_models.py --out_path='both_models_2048_4000.csv' --mingru_dataset_path='mingru_2048_4000_001.pt' --transformer_dataset_path='transformer_2048_4000_001.pt'
