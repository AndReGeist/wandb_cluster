#!/bin/bash
#SBATCH --job-name=BASICEXAMPLE
#SBATCH --output=.out/BASICEXAMPLE_OUTPUT.out
#SBATCH --error=.out/BASICEXAMPLE_ERROR.err
#SBATCH --time=00:10:00
#SBATCH --mem=1G

### The last part consists of regular shell commands:
### Change to working directory
cd /home/usr/workingdirectory

### Execute your application
python3 basic_example.py --seed 1234