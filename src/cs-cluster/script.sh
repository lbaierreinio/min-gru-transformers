#!/bin/bash
source /w/340/lucbr/miniconda3/bin/activate
export PYTHONPATH="$PYTHONPATH:/w/331/lucbr/min-gru-transformers/src"
conda activate pytorch_env
python3 experiments/train_model.py --dataset_path='mingru_d_100_2000.pt' --model=0
#python3 experiments/evaluate_model.py --validation_dataset_path='exp_96.pt' --model_in_path='mingru_exp_64_model.pt' --out_path='exp_64_96.csv'