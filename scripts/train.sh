#!/bin/bash
#SBATCH --job-name=vits            # job name (shows in squeue)
#SBATCH --partition=scavenger      # group of nodes that has GPUs
#SBATCH --gres=gpu:1               # 1 GPU
#SBATCH --cpus-per-task=4          # number of cores
#SBATCH --mem=32G                  # memory
#SBATCH --time=24:00:00            # walltime (max run time)
#SBATCH --output=logs/%x_%j.out    # output file, %j = job ID
#SBATCH --error=logs/%x_%j.err     # error file

# Usage (from ~/vits):   sbatch train.sh ljs_1000.json run_1000
#   $1 = config file in configs/     $2 = run name (folder in logs/)

module load miniconda3             # load packages
source activate vits
cd ~/vits
python train.py -c configs/$1 -m $2   # the final command
