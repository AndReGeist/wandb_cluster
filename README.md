Research on optimizing ML models with many parameters is **significantly more efficient** if we...
- Offload computation to a cluster, to be able to scale computation
- Automize logging and analyzing experiments using e.g. weights and biases

This repository provides Python and Shell script to run experiments on a HPC cluster using [wandb](https://wandb.ai/site) to log and analyze experiments.
As ML example, Neural ODE regression is chosen using JAX, Diffrax, Equinox, and Optax.

# Weights and biases
[Weights and biases](https://wandb.ai/site), aka wandb, is a platform for ML research that allows to:
- Track experiments
- Visualize the training process online (e.g. objective functions and gradients)
- Share results with a team
- Log data sets
- Automize hyper-parameter search

[Click here](https://theaisummer.com/weights-and-biases-tutorial/) for a brief introduction to wandb. To use wandb you need to:
1) Create a free account (e.g. with an academic license)
2) Install the wandb library using pip
3) Specify the experiment hyperparameters in your python script using a simple command from wandb as well as store your loss using another wandb function.
4) Run the experiment. Open your wandb account in a browser and marvel at all the shiny plots of your experiments.

In addition, wandb allows to tune hyperparameters either with random-, grid-, or Bayes- search using so called [sweeps](https://docs.wandb.ai/guides/sweeps).

With the above functionality of wandb, I setup **two code examples** (NeuralODE regression with JAX) that cover the following use-cases:
1) You **specify the optimization hyperparameters in an excel**. The head of the excel specifies the parameter name and the $n$-th row specifies the $n$-th experiment parameters. When calling your Python script, you specify which row of the excel is used to create the config file.
2) You **use a wandb sweep to do hyperparameter tuning** for you.

In the above use cases,  I also used wandb's [data and model versioning functionality](https://theaisummer.com/weights-and-biases-tutorial/)

To get started with wandb, I created a [Google Colab]() using wandb to document the training of a neural ODE (using JAX, Equinox, Diffrax, and Optax). In particular, I use wandb to:
- Log a dataset after creation on the wandb server
- Load a data set before training from the wandb server
- Log the training configurations
- Log the optimization results
- Log the model parameters and gradient steps

<img src="https://github.com/AndReGeist/wandb_cluster_neuralode/blob/main/images/Pasted%20image%2020230322135007.png" width="50%" height="50%">
<img src="https://github.com/AndReGeist/wandb_cluster_neuralode/blob/main/images/Pasted%20image%2020230322135007.png" width="50%" height="50%">

After playing around with the code, you can download the [python files and shell scripts]() and move on to using the HPC cluster.

# HPC Cluster
For the above use-cases, I want to speed up and parallize computation using a high performance compute cluster.

For using the cluster (in my case the cluster of the RWTH Aachen), I found the following ressources helpful:
- [Introduction to linux with nice Youtube videos](https://hpc-wiki.info/hpc/Introduction_to_Linux_in_HPC)
- [RWTH high performance computing](https://help.itc.rwth-aachen.de/service/rhr4fjjutttf/)

To use the RWTH cluster you need to:
1) Create an RWTH HPC account in [RegApp](https://regapp.itc.rwth-aachen.de) 
2) Connect to the [cluster](https://help.itc.rwth-aachen.de/service/rhr4fjjutttf/article/b3027aeb8fd64f3d853e8ce70fbcfbe7/) via
	- VPN (via password authentication or more conviniently using an [SSH key](https://hpc-wiki.info/hpc/Introduction_to_Linux_in_HPC/SSH_Connections))
	- Desktop client
3) Write a [shell script](https://hpc-wiki.info/hpc/Introduction_to_Linux_in_HPC/Shell_scripting) that instructs [SLURM](https://hpc-wiki.info/hpc/SLURM) what you want the cluster to do.

For the afore mentioned usecases,  I wrote two shell scripts:
1) A script for the **first use case** which uses SLURM [job arrays](https://hpc-wiki.info/hpc/SLURM#Array%20and%20Chain%20Jobs) to tell the cluster that we want to run the python script for all parameter settings (specified by the rows of the excel) **in parallel**.
2) A script for the **second use case** which uses SLURM [job arrays](https://hpc-wiki.info/hpc/SLURM#Array%20and%20Chain%20Jobs) to split the potentially huge job into smaller jobs that are executed **one after the other**. This is necessary, as automatic parameter tuning via *sweep* might take a long time and to the best of my knowledge cannot be parallelized. So after a specified amount of time, SLURM saves the current program state and then starts a new job starting with the stored program state.

A benefit of using wandb is that we do not need to [transfer files](https://www.youtube.com/watch?v=gOYsbBxKXas) after optimization from the cluster to our local computer.

After connecting to the cluster, run the following commands to start a job:
1) Get git repo `...`
2) Install libraries `...`
3) Execute shell script `...`
