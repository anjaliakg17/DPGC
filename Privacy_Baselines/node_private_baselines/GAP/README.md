
## Requirements

This code is implemented in Python 3.9 using PyTorch-Geometric 2.1.0 and PyTorch 1.12.1.
Refer to [requiresments.txt](./requirements.txt) to see the full list of dependencies.

## Notes
1. The code includes a custom C++ operator for faster edge sampling required for the node-level DP methods. PyTorch will automatically build the C++ code at runtime, but you need to have a C++ compiler installed (usually it is handled automatically if you use conda).

2. We use [Weights & Biases](https://docs.wandb.ai/) (WandB) to track the training progress and log experiment results. To replicate the results of the paper as described in the following, you need to have a WandB account. Otherwise, if you just want to train and evaluate the model, a WandB account is not required.

4. We use [Dask](https://jobqueue.dask.org/) to parallelize running multiple experiments on high-performance computing clusters (e.g., SGE, SLURM, etc). If you don't have access to a cluster, you can also simply run the experiments sequentially on your machine (see [usage section](#usage) below).

3. The code requires autodp version 0.2.1b or later. You can install the latest version directly from the [GitHub repository](https://github.com/yuxiangw/autodp) using: 
    ```
    pip install git+https://github.com/yuxiangw/autodp
    ```


## Usage

### Replicating the paper's results
To reproduce the paper's results, please follow the below steps:  

1. Set your WandB username in [wandb.yaml](./wandb.yaml) (line 7). This is required to log the results to your WandB account.

2. Execute the following python script:
    ```
    python experiments.py --generate
    ```
    This creates the file "jobs/experiments.sh" containing the commands to run all the experiments.

3. If you want to run the experiments on your own machine, run:
    ```
    sh jobs/experiments.sh
    ``` 
    This trains all the models required for the experiments one by one. Otherwise, if you have access to a [supported HPC cluster](https://jobqueue.dask.org/en/latest/api.html), first configure your cluster setting in [dask.yaml](./dask.yaml) according to Dask-Jobqueue's [documentation](https://jobqueue.dask.org/en/latest/configuration.html). Then, run the following command:
    ```
    python experiments.py --run --scheduler <scheduler>
    ```
    where `<scheduler>` is the name of your scheduler (e.g., `sge`, `slurm`, etc). The above command will submit all the jobs to your cluster and run them in parallel. 
    

  4. Use [results.ipynb](./results.ipynb) notebook to visualize the results as shown in the paper. Note that we used the [Linux Libertine](https://libertine-fonts.org/) font in the figures, so you either need to have this font installed or change the font in the notebook.

### Training individual models

