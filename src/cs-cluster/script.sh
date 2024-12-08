#!/bin/bash
source /w/340/lucbr/miniconda3/bin/activate
export PYTHONPATH="$PYTHONPATH:/w/331/lucbr/min-gru-transformers/src"
conda activate pytorch_env
python3 -u experiments/evaluate_model.py --model_path='mingru_1536_2560_final_m.pt' --dataset_path='mingru_d_4096_1000_hard.pt'
