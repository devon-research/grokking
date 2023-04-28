#!/bin/bash
#SBATCH --job-name=grokking      # a name for the job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=10G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:30:00          # total run time limit (HH:MM:SS)

module purge
module load anaconda3/2022.5
conda activate grokking

python scripts/train.py