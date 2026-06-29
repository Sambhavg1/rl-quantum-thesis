#!/bin/bash
#SBATCH --job-name=AME46_SCALE
#SBATCH --output=ame46_scaled_%j.out
#SBATCH --error=ame46_scaled_err_%j.log
#SBATCH --time=24:00:00   # <--- BUMPED TO 24 HOURS
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16 # <--- Sweet spot for single-env NumPy
#SBATCH --mem=32G          # <--- Plenty of overhead space
#SBATCH --gres=gpu:1       # <--- Easily handles the deeper 2048->512 network

cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTRAOP_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTEROP_THREADS=2

PYTHON_PATH="/home/sambhav/miniconda3/envs/quantum/bin/python"

echo "=========================================="
echo "Starting AME(4,6) Scaled VAST Search..."
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Horizon: 200 steps"
echo "=========================================="

$PYTHON_PATH ame46_pudda_vast_scaled.py

echo "=========================================="
echo "Job finished at $(date)"
echo "=========================================="