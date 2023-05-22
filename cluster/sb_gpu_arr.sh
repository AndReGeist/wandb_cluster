#!/usr/bin/zsh

#SBATCH --job-name=JOB_ARRAY_GPU
#SBATCH --array=1-20%20
#SBATCH --output=./outputs/%J.txt
#SBATCH --error=./errors/%J.txt
#SBATCH --time=00:50:00
#SBATCH --mem=8G
#SBATCH --gres=gpu:1

### Conda
export CONDA_ROOT=$HOME/miniconda3
. $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"

### Cuda
module load CUDA/11.8.0

# Print debug info
echo; export; echo; nvidia-smi; echo

### Begin of executable commands
conda activate paramter_tow

export XLA_FLAGS=--xla_gpu_cuda_data_dir=/cvmfs/software.hpc.rwth.de/Linux/RH8/x86_64/intel/skylake_avx512/software/CUDA/11.8.0

"$@"
