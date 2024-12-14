# Evaluating Min GRU vs. Transformers on Long-Context Reasoning and Question Answering

## Prerequisites
- Conda

## Steps to Run
1) Clone the repository onto your machine.
2) Create the virtual environment: ```conda env create -f environment.yml```
3) Activate the environment: ```conda activate pytorch_env```
4) Ensure the src directory is in your Python Path: ```export PYTHONPATH="$PYTHONPATH:/path/to/repository/src"```
> **Note:** This will only add the src directory to your path for your current session.  Write the export command to your .bashrc, .zshrc, or equivalent to add the path permanently.
5) Run test cases: ```pytest```
6) If running SQuAD v2 evaluation, set `export HF_HOME="/path/to/another/directory/datasets"` to set the cache directory (in particular, if working on the UofT comps server, downloading the dataset on your default home directory can use up a lot of your quota).


## Synthetic Experiments

All synthetic experiments were run on the University of Toronto's SLURM cluster. Example scripts to execute these experiments are located under the `cs-cluster` directory.

### Table of Contents
1. [Generate a Synthetic Dataset](#generate-a-synthetic-dataset)
2. [Train a Model](#train-a-model)
3. [Evaluate a Model](#evaluate-a-model)
4. [Profile Memory and Runtime](#profile-memory-and-runtime)
5. [Train Both Models](#train-both-models)

---

### 1. Generate a Synthetic Dataset

Use the script `generate_dataset.py` to generate synthetic datasets for the Transformer and MinGRU models.

**Script Location:**
```
min-gru-transformers/src/datasets/synthetic/generate_dataset.py
```

**Usage:**
```bash
generate_dataset.py --dataset_path=dataset.pt
```

**Saves:**
- MinGRU dataset as `mingru_dataset.pt`
- Transformer dataset as `transformer_dataset.pt`

You can tweak the dataset parameters using the `DatasetConfig` dataclasses defined within the script.

---

### 2. Train a Model

Train a MinGRU or Transformer model on a dataset using `train_model.py`.

**Script Location:**
```
min-gru-transformers/src/experiments/train_model.py
```

**Usage:**
```bash
train_model.py --dataset_path=dataset_path.pt --out_path=out.csv --model=(0/1)
```

- `--dataset_path`: Path to a valid `.pt` dataset file.
- `--out_path`: Path to save detailed training results (CSV format).
- `--model`: Specify `0` for MinGRU or `1` for Transformer.

**Saves:**
- MinGRU model as `mingru.pt`
- Transformer model as `transformer.pt`
- Loss and Accuracy graphs as `loss.png` and `accuracy.png`

You can customize model and training parameters using the `TransformerConfig`, `MinGRUConfig`, and `TrainConfig` dataclasses defined within the script.

---

### 3. Evaluate a Model

Evaluate a trained model's performance on a dataset using `evaluate_model.py`.

**Script Location:**
```
min-gru-transformers/src/experiments/evaluate_model.py
```

**Usage:**
```bash
evaluate_model.py --dataset_path=dataset_path.pt --model_path=model_path.pt
```

- `--dataset_path`: Path to a valid `.pt` dataset file.
- `--model_path`: Path to a valid `.pt` model file.

**Output:**
- The loss and accuracy will be printed to `stdout`.

---

### 4. Profile Memory and Runtime

Compare the memory and runtime of training both model architectures across varying sequence lengths using `profile_train_models.py`.

**Script Location:**
```
min-gru-transformers/src/experiments/profile_train_models.py
```

**Usage:**
```bash
profile_train_models.py
```

You can customize configurations (e.g., training parameters, sequence lengths, and model architectures) using the `MinGRUConfig`, `TransformerConfig`, and `ProfileTrainConfig` dataclasses defined in the script.

**Output:**
- `memory_epochs.png`: Memory usage over epochs.
- `time_epochs.png`: Training time over epochs.

---

### 5. Train Both Models

Train both the Transformer and MinGRU models simultaneously using `train_models.py`.

**Script Location:**
```
min-gru-transformers/src/experiments/train_models.py
```

**Usage:**
```bash
train_models.py --transformer_dataset_path=transformer_dataset_path.pt --mingru_dataset_path=mingru_dataset_path.pt --out_path=out_path.csv
```

- `--transformer_dataset_path`: Path to the Transformer dataset.
- `--mingru_dataset_path`: Path to the MinGRU dataset.
- `--out_path`: Path to save detailed results (CSV format).

**Output:**
- Training results saved to `out_path.csv`.
- Accuracy and loss graphs:
  - `loss.png`
  - `accuracy.png`
- Model files:
  - `mingru.pt`
  - `transformer.pt`

You can tweak training parameters and model architectures using the dataclasses defined in the script.

---

## SQuAD
The training and evaluation on SQuAD is available in one command via
```bash
python train_squad.py
```

All hyperparameter configurations lie at the top of the script. Flags such as `use_compile` or `ampere_gpu` must be set appropriately in the script depending on the type of GPU.