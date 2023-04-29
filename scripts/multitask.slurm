#!/bin/bash
#SBATCH --job-name=grokking      # a name for the job
#SBATCH --ntasks=4               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=16G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:20:00          # total run time limit (HH:MM:SS)

module purge
module load anaconda3/2023.3
module load julia/1.8.2
conda activate grokking

julia scripts/experiments.jl