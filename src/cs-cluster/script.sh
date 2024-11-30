#!/bin/bash
source /w/340/lucbr/miniconda3/bin/activate
export PYTHONPATH="$PYTHONPATH:/u/lucbr/min-gru-transformers/src"
conda activate pytorch_env
python3 experiments/transformer_synthetic.py --train_dataset_path='even_128.pt' --validation_dataset_path='odd_256.pt' --out_path='out_even_odd.csv'