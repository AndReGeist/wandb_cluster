#!/usr/bin/zsh

#SBATCH --job-name=BASIC_EXAMPLE_GPU_ARRAY
### Run in total 30 jobs á 10 minutes with max 5 jobs at a time
#SBATCH --array=1-30%5
#SBATCH --output=./outputs/%J.txt
#SBATCH --error=./errors/%J.txt
#SBATCH --time=00:10:00
#SBATCH --mem=5G
#SBATCH --gres=gpu:volta:1

### Conda
export CONDA_ROOT=$HOME/miniconda3
. $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"

### Begin of executable commands
conda activate diffrax

"$@"
