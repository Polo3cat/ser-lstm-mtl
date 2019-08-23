#!/bin/bash
#SBATCH -J batch_A
#SBATCH -c 8
#SBATCH --gres=gpu:2

srun trainAbase.sh &
srun trainAmtl.sh &
srun trainAprivate &
srun trainAprivate2
