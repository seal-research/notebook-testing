# Artifact for ASE 2025 submission - NBTest

The following is a very brief outline of the steps required to run NBTest.

Please note that we will refine this README file for the final artifact submission, explaining each step in more detail.

## Installation

1. Download/Clone this repository and navigate to it

2. Install Conda by following the instructions in this documentation (if not already installed): https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions

3. Run the following commands to setup the conda environment.

```sh
conda env create -f env.yml
conda activate nbtest_env
pip install -e ./nbtest_plugin/
```

## Usage

### Results extraction

The `scripts/run.sh` bash script, for a given input Jupyter notebook specified by `nb_path`, executes the following stages:
1. Assertion Generation (NBTest-gen)
2. Testing the generated notebook (NBTest-plugin)
3. Mutation testing 
4. Kaggle Versions evaluation 

and saves all the results in the `result_dir`. 

Execute the following command from the root of this repository, to get the results for a given notebook. 

```sh
$ ./scripts/run.sh $path_to_notebook $dynamic_iteration $stat_method $conf_level $pytest_ite $mutest_ite $result_dir`
```

For example,
```
$ ./scripts/run.sh ./sample.ipynb 10 "chebyshev" 0.99 10 10 ./results`

```

### Using NBTest-lab-extension 

Install `NBTest-lab-extension`, using

```sh
pip install -e ./nbtest_lab_extension
```

## Results
- The Kaggle jupyter notebook we have considered in our paper are in `./data/kaggle_notebook.csv`. We use the notebooks whose `executable` column is 1 in our experiments.
- The url for versions for each notebook we have considers is in `./data/notebook_versions.csv`