# Evaluating Min GRU vs. Transformers on Long-Context Retrieval

### Prerequisites
- Currently, we are using conda with pip to manage our dependencies. See (https://docs.anaconda.com/miniconda/) for installation details. Note that we may end up using venv instead, as this is what is recommended on the U of T servers. Also, the use of Pip & Conda together is sometimes discouraged, which is another reason to switch to venv.

### Steps to Run
1) Clone the repository onto your machine.
2) Create the virtual environment: ```conda env create -f environment.yml```
3) Activate the environment: ```conda activate pytorch_env```
4) Ensure the project is in your Python Path: ```export PYTHONPATH="$PYTHONPATH:/path/to/repository```
5) Run test cases: ```pytest```

