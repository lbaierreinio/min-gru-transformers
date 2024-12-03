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