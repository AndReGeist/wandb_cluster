Research on optimizing ML models with many parameters is **significantly more efficient** if we...
- Automate logging, analyzing experiments, and parameter search using e.g. weights and biases
- Offload computation to a compute cluster, to be able to scale computation

This repository provides Python and Shell script to run experiments on a HPC cluster using [wandb](https://wandb.ai/site) to log and analyze experiments.
As ML example, supervised neural ODE regression is chosen using JAX, Diffrax, Equinox, and Optax.

For a brief introduction to using W&B on a cluster, I recommend the following [video](https://www.youtube.com/watch?v=LRmnr3LMS-4).

# Weights and biases
[Weights and biases](https://wandb.ai/site), aka W&B, is a platform for ML research that allows to:
- Track experiments (configurations, objective values)
- Visualize the training process online (e.g. objective functions and gradients)
- Track data sets and models
- Automate parameter search
- Share results with a team

[Click here](https://theaisummer.com/weights-and-biases-tutorial/) for a brief introduction to wandb. To use wandb you need to:
1) Create a free account (e.g. with an academic license)
2) Install the wandb library using pip
3) Specify the experiment hyperparameters in your python script using a simple command from wandb as well as store your loss using another wandb function.
4) Run the experiment. Open your wandb account in a browser and marvel at all the shiny plots of your experiments.

To get started with wandb, I created a [Google Colab](https://colab.research.google.com/github/AndReGeist/wandb_cluster_neuralode/blob/main/colab/basic_example.ipynb#scrollTo=qXAp1blGadTl). In this example, wandb is used to:
- Log a dataset after creation on the wandb server
- Load a data set before training from the wandb server
- Log the training configurations
- Log the optimization results
- Log the model parameters and gradient steps
- Automate parameter search with W&B sweeps

After playing around with the code, you can download the python files and shell scripts in the folder `cluster` and move on to using the HPC cluster.

# HPC Cluster
A cluster enables us to scale computation if needed.

For using the cluster (in my case the cluster of the RWTH Aachen), I found the following ressources helpful:
- [Introduction to linux with nice Youtube videos](https://hpc-wiki.info/hpc/Introduction_to_Linux_in_HPC), you should feel comfortable using console commands and VIM
- [Getting started with HPC](https://hpc-wiki.info/hpc/Getting_Started)
- [Intro to job scheduling](https://pdc-support.github.io/hpc-intro/09-scheduling/)
- [RWTH HPC mainpage](https://help.itc.rwth-aachen.de/service/rhr4fjjutttf/)
- [RWTH cluster - slurm commands](https://help.itc.rwth-aachen.de/en/service/rhr4fjjutttf/article/3d20a87835db4569ad9094d91874e2b4/)
- [RWTH cluster - account infos](https://help.itc.rwth-aachen.de/service/rhr4fjjutttf/article/23ef5b95361d4007836d7315618dbed9/)
- [RWTH cluster - Batch script examples](https://help.itc.rwth-aachen.de/service/rhr4fjjutttf/article/6e4d3ad2573d4e41a5fab9b65dbd320a/)

The folder `cluster` contains the following files:
- `basic_example.py` - Contains the same code as the notebook. Here we want to run the functions `main()` or `create_dataset`.
- `sb.sh` - A simple shell script that activates a conda environment and executes a command such as executing a Python script.
- `sweep_config.yaml` - Contains the settings for the W&B sweep.

In what follows, we do the following jobs on the cluster:
1) Run the script `basic_example.py` using python-fire to conveniently assign input parameters through the console
2) Setup a W&B sweep to do automatic parameter search on the cluster and manually assign compute nodes on the cluster on which the W&B agent executes `basic_example.py`
3) Basically the above step, but we use a "job array" to run many jobs for the W&B sweep in parallel. 

**Useful console commands:**
- Check job status of user `squeue -u <user_id> --start`
- Check specific job status `squeue --job <job_id>`
- Get job efficiency report `seff JOBID`
- Cancel job `scancel <job_id>`
- Get info on [job history](https://slurm.schedmd.com/sacct.html) `sacct`
- Get manual on sbatch `man sbatch`
- Get partition information `sinfo -s`
- Check consumed core hours ``r_wlm_usage -p <cluster-project-name> -q``

## Set up your login node
To use a HPC cluster you need to:
1) Create an HPC account, e.g. for the RWTH cluster via [RegApp](https://regapp.itc.rwth-aachen.de) 
2) Optional: Setup cluster login using [SSH key](https://hpc-wiki.info/hpc/Introduction_to_Linux_in_HPC/SSH_Connections)
	1) Setup ssh key
		1) Navigate to `.ssh` in home directory
		2) Create key `ssh-keygen` and specify PW
	2) Send key to cluster `ssh-copy-id -i <public-key-name> <user-name>@login18-1.hpc.itc.rwth-aachen.de`
3) Optional: Create `config` file in `.ssh` with SSH preset:
			   `host <short-name> HostName <host-name> User <username> TCPKeepAlive yes ForwardX11 yes`
4) Write a [shell script](https://hpc-wiki.info/hpc/Introduction_to_Linux_in_HPC/Shell_scripting) that instructs [SLURM](https://hpc-wiki.info/hpc/SLURM) what you want the cluster's **compute nodes** to do. Check [scheduling basics](https://hpc-wiki.info/hpc/Scheduling_Basics).
	- `--mem=<memlimit>` must not be used on RWTH clusters
	- Shebang of the batch script **must be** `#!/usr/bin/zsh` on RWTH cluster
5) Clone project from git `git clone <git-project-URL>`
	- If authentication fails see [here](https://ginnyfahs.medium.com/github-error-authentication-failed-from-command-line-3a545bfd0ca8)
6) Setup environment using conda:
	- [using conda on RWTH cluster](https://help.itc.rwth-aachen.de/en/service/rhr4fjjutttf/article/960a597fa06e426ba304275a7584f8c8/#5.4)
	- [conda tutorial by Princeton University](https://researchcomputing.princeton.edu/support/knowledge-base/python)
	1) Install miniconda on login node
	   `$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`
	   `$ bash Miniconda3-latest-Linux-x86_64.sh`
	   `$ conda config --set auto_activate_base false`
	   2) If `conda` command is not found execute `export PATH="/home/<username>/miniconda/bin:$PATH"` with your username.
	   3) Install conda environment from yaml file `conda env create --file conda_env.yaml`
	   4) Activate environment `conda activate <environment name>`

## Start jobs
In a console window (Linux/ Mac OS) run the following commands to start a job:
1) Connect to the [cluster](https://help.itc.rwth-aachen.de/service/rhr4fjjutttf/article/b3027aeb8fd64f3d853e8ce70fbcfbe7/) **login node** (after connecting to the RWTH vpn) via 
   `ssh <username>@login18-1.hpc.itc.rwth-aachen.de`
2) Start a [screen session](https://linuxize.com/post/how-to-use-linux-screen/) which keeps your linux session running in case your SSH disconnects:
	- Start session `screen -S <session-name>`
	- Reattach to linux screen 10386 `screen -r 10386`
	- List running screen sessions `screen -ls`
	- List all windows `<Ctrl+a> "`
	- Close the current region `<Ctrl+a> X`
	- Detach from screen session `<Ctrl+a> d`
3) Activate conda environment `conda activate <env>` and login into wandb `wandb login` 
4) Execute a shell script `sbatch sb.sh <python-call> <script-input parameters>`
e.g. 
`sbatch sb.sh python3 basic_example.py --seed 123`
`sbatch sb.sh python3 -c 'import basic_example; create_dataset(seed_123)'` 
5) Optional: [Transfer files](https://hpc-wiki.info/hpc/File_Transfer) from cluster node to local computer
	- Using secure copy for single files, e.g. `$ scp your_username@remotehost.edu:foobar.txt /some/local/directory`
	- Using **rsync** for multiple files

**W&B sweeps**

To let W&B do the parameter search for you...
1) Open W&B in your browser, navigate to `Projects / <project name> / Sweeps` and click on "Create Sweep"
2) Copy "sweep_config.yaml" (in the repo folder `cluster`) and paste it into the browser. Click on "Intialize sweep"
3) In the console execute
   ```
   sbatch sb.sh wandb agent <name-of-sweep-as-in-wand-browser>
   ```
as often as you want. W&B will do the parameter selection and tell the compute nodes what to run. If you want to run many jobs at ones, it is better to use SLURM job arrays as detailed below.

**Running W&B sweeps using SLURM job arrays**

A sweep might run for a long time, depending on how many parameters shall be checked. If a job is too long, it takes more time until the job gets a free slot on the cluster and potentially costs more cluster credits. We avoid these issues by using SLURM [job arrays](https://hpc-wiki.info/hpc/SLURM#Array_and_Chain_Jobs) in the shell script `sb_gpu_arr.sh`. Here, we run in total 30 jobs á 10 minutes with 5 jobs running at the same time by using the command...
```
#SBATCH --array=1-30%5
```
Starting a wandb sweep with SLURM job arrays is the same as before while the sbatch command slightly changed... 
```
sbatch sb_gpu_arr.sh wandb agent <name-of-sweep-as-in-wand-browser>`
```

For a **very long simulation**, you can use the shell command to tell SLURM that you want to run 30 jobs after each other...
```
#SBATCH --array=1-30%1
```
...and call the shell script as follows...
```
sbatch sb_arr_altered.sh python3 <simulation-file-name> -statefile=simstate.state
```
If the simulation does not finish in time it will be followed by the next array task, which picks up right at where the simulation left (by reading in "simstate.state").

## Handy tricks
- The python-fire library automatically generates command line interfaces (CLIs) from python objects, e.g. assume you have a Python file "main.py"...
```python
def main(batch_size=32, seed=5678):
	"""some program"""
		
if __name__ == '__main__':  
    fire.Fire(main)
```
... then you can run main from the console via...
```python
python main.py --seed=42
```

- You can open a second console and connect to the same remote computer. Then, the console command `top` provides a dynamic real-time view of the system.

- The only way for Slurm to detect success or failure of running the Python program is the exit code of your job script. You can store the exitcode after executing the program to prevent it from being overwritten...
```
#!/bin/bash
#SBATCH …
myScientificProgram …
EXITCODE=$?
cp resultfile $HOME/jobresults/
/any/other/job/closure/cleanup/commands …
exit $EXITCODE
```
