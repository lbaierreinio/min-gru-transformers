#!/bin/bash
source /w/340/lucbr/miniconda3/bin/activate
export PYTHONPATH="$PYTHONPATH:/u/lucbr/min-gru-transformers/src"
conda activate pytorch_env
python3 experiments/transformer_synthetic.py --dataset_path='dataset.pt' --out_path='out.csv'