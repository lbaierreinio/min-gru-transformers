#!/bin/bash
source /w/340/lucbr/miniconda3/bin/activate
export PYTHONPATH="$PYTHONPATH:/w/331/lucbr/min-gru-transformers/src"
conda activate pytorch_env
python3 -u experiments/train_model.py --dataset_path='transformer_1536_2560_final.pt' --out_path='transformer_1536_2560_final.csv' --model=1
