#!/bin/bash
#SBATCH --job-name=AME_1NC
#SBATCH --output=ame_1nc_%j.out
#SBATCH --error=ame_1nc_err_%j.log
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

cd "$SLURM_SUBMIT_DIR"

export TF_FORCE_GPU_ALLOW_GROWTH=true
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTRAOP_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTEROP_THREADS=2

PYTHON_PATH="/home/sambhav/miniconda3/envs/quantum/bin/python"

echo "Starting 1 Non-Clifford Gate Search (No Penalty)..."
$PYTHON_PATH ame_1nc_without_penalty.py
echo "Job finished."