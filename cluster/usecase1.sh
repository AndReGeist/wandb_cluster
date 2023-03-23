lOB#!/usr/local_rwth/bin/zsh


### As array

#SBATCH --array=0-11


### Name the job

#SBATCH --job-name=MC_Simulation_Reset_Time


### declare the merged STDOUT/STDERR file

#SBATCH --output=results/outputs/output.%J.txt


### Memory your job needs per node, e. g. 1 GB

#SBATCH --mem=3G


### Time limit in hh:mm:ss

#SBATCH --time=23:00:00


# Insert this after any #SLURM commands

export CONDA_ROOT=$HOME/miniconda3

. $CONDA_ROOT/etc/profile.d/conda.sh

export PATH="$CONDA_ROOT/bin:$PATH"

# but naturally before using any python scripts


# define and create a unique scratch directory

SCRATCH_DIRECTORY=/home/pb374595/hpcwork/00_DSME_Projects/01_event_based_BO/event-triggered_BO

mkdir -p ${SCRATCH_DIRECTORY}

cd ${SCRATCH_DIRECTORY} || return


# Activate conda env

source /home/pb374595/.zshrc

conda activate event_triggered_BO


# we execute the job and time it

time python3 run_MC_sim_expected_reset_time.py -p ${SCRATCH_DIRECTORY} --log_name $SLURM_ARRAY_JOB_ID --run_id $SLURM_ARRAY_TASK_ID --start_seed 0 --end_seed 10000 -T 400 --epsilons 0.005 0.01 0.05 0.1 0.5 0.7 --deltas 0.01 0.1
