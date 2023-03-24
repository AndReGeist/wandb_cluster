#!/usr/bin/zsh

#SBATCH --job-name=BASIC_EXAMPLE
#SBATCH --output=./outputs/%J.txt
#SBATCH --error=./errors/%J.txt
#SBATCH --time=00:10:00
#SBATCH --mem=5G

### Conda
export CONDA_ROOT=$HOME/miniconda3
. $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"

### Begin of executable commands
conda activate diffrax

"$@"
