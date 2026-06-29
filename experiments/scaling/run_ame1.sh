#!/bin/bash
#SBATCH --job-name=AME_Search_GPU
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1            # Request 1 GPU
#SBATCH --time=12:00:00         # AME search takes time
#SBATCH --output=logs/ame_%j.log
#SBATCH --error=logs/ame_err_%j.log

cd $SLURM_SUBMIT_DIR

# Using the Bulbasaur Profile python path directly
# to avoid conda activate issues
PYTHON_PATH="/home/sambhav/miniconda3/envs/quantum/bin/python"

echo "Starting AME Search on Bulbasaur..."
echo "Date: $(date)"

$PYTHON_PATH ame_gpu.py

echo "Job Finished at $(date)"