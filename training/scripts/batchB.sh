#!/bin/bash
#SBATCH -J batch_B
#SBATCH -c 8
#SBATCH --gres=gpu:2

srun trainBbase.sh &
srun trainBmtl.sh &
srun trainBprivate &
srun trainBprivate2
