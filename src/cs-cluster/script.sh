#!/bin/bash
source /w/340/lucbr/miniconda3/bin/activate
export PYTHONPATH="$PYTHONPATH:/w/331/lucbr/min-gru-transformers/src"
conda activate pytorch_env
python3 -u experiments/train_model.py --dataset_path='transformer_128_512_d_easy_grammar.pt' --out_path='transformer_d_128_512_8000_easy_grammar.csv' --model=1
