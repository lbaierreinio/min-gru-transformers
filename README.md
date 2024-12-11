# Evaluating Min GRU vs. Transformers on Long-Context Retrieval

### Prerequisites
- We are using conda to manage our dependencies. See (https://docs.anaconda.com/miniconda/) for installation details. Note that we may switch to venv later, since this is what is recommended on the U of T servers.

### Steps to Run
1) Clone the repository onto your machine.
2) Create the virtual environment: ```conda env create -f environment.yml```
3) Activate the environment: ```conda activate pytorch_env```
4) Ensure the src directory is in your Python Path: ```export PYTHONPATH="$PYTHONPATH:/path/to/repository/src"```
> **Note:** This will only add the src directory to your path for your current session.  Write the export command to your .bashrc, .zshrc, or equivalent to add the path permanently.
5) Run test cases: ```pytest```
6) If running SQuAD v2 evaluation, set `export HF_HOME="/path/to/another/directory/datasets"` to set the cache directory (in particular, if working on the UofT comps server, downloading the dataset on your default home directory can use up a lot of your quota).


### Running Synthetic Experiments
Note that all synthetic experiments were run on the University of Toronto's SLURM cluster. See example scripts under the `cs-cluster` directory to run the scripts on the SLURM cluster.
1) Generate a Synthetic Dataset using the following script (found under `min-gru-transformers/src/datasets/synthetic`):

 > `generate_dataset.py --dataset_path=dataset.pt`. 

> This will output two datasets, one for the Transformer Model, and one for the MinGRU model. The MinGRU dataset will be saved at `mingru_dataset_path.pt`, and the Transformer dataset will be saved at `transformer_dataset_path.pt`. You may tweak the parameters of the synthetic dataset using the `DatasetConfig` dataclasses defined within the script.

2) Train a model on a dataset using the following script (found under `min-gru-transformers/src/experiments`):

> `train_model.py --dataset_path=dataset_path.pt --out_path=out.csv --model=(0/1)`.

> The `dataset_path` argument must point to a valid .pt file. The `out_path` will store detailed results of the training run. The `model` argument dictates whether the model is a MinGRU model (0) or a Transformer model (1). If the MinGRU model is selected, the final model will be saved under `mingru.pt`, and if the Transformer model is selected, the final model will be saved under `transformer.pt`. You may tweak the parameters of each model, and training configuration, using the `TransformerConfig`, `MinGRUConfig`, and `TrainConfig` dataclasses defined within the script.

3) Evaluate the model's performance on a specified dataset using the following script (found under `min-gru-transformers/src/experiments`):

> `evaluate_model.py --dataset_path=dataset_path.pt --model_path=model_path.pt`

> The `dataset_path` and `model_path` must both point to valid .pt files. The loss and accuracy on the dataset will be printed to `stdout`.

4) Compare the memory and runtime of training both model architectures on varying sequence lengths using the following script (found under `min-gru-transformers/src/experiments`):

> `profile_models.py`

> You may tweak the training configurations, number of sequence lengths that are tested, and MinGRU/Transformer model architectures using the `MinGRUConfig`, `TransformerConfig`, and `ProfileTrainConfig` dataclasses defined within the script. The results will be saved to 2 graphs, memory_epochs.png and time_epochs.png.

5) Evaluate the Transformer & MinGRU model on a series of different datasets using the following script (found under `min-gru-transformers/src/experiments`).

 > `train_model_multiple_datasets.py --out_path=out_path.csv`

> This is the most comprehensive test and will compare the results of the MinGRU and Transformer model on a variety of different generations of the synthetic dataset. You may tweak the parameters of the training dataset, number of datasets tested, MinGRU and Transformer models, and training configuration in the dataclasses defined in the script. Detailed results in CSV form will be saved to `out_path`. Two PNG files, {seq_len}_loss.png and {seq_len}_accuracy.png will be saved for each value of `seq_len` in the test, which plot loss and accuracy of the MinGRU and Transformer model over each epoch of training on the current dataset. Finally, two PNGs, all_seq_len_accuracy.png and all_seq_len_losses.png will graph all of the results for all of the sequence lengths.

