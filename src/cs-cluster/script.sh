#!/bin/bash
source /w/340/lucbr/miniconda3/bin/activate
export PYTHONPATH="$PYTHONPATH:/w/331/lucbr/min-gru-transformers/src"
conda activate pytorch_env
python3 -u experiments/train_models.py --out_path='both_models_1024_4000_final.csv' --mingru_dataset_path='mingru_1024_4000_final.pt' --transformer_dataset_path='transformer_1024_4000_final.pt'
