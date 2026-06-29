#!/bin/bash
#SBATCH --job-name=N4_APPROX
#SBATCH --output=n4_approx_%j.out
#SBATCH --error=n4_approx_err_%j.log
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

cd "$SLURM_SUBMIT_DIR"

# Optimize matrix operations and TensorFlow threading
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTRAOP_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTEROP_THREADS=2

# Path to your quantum conda environment
PYTHON_PATH="/home/sambhav/miniconda3/envs/quantum/bin/python"

echo "=========================================="
echo "Starting N=4 Fractional Gate Approximation..."
echo "Job ID: $SLURM_JOB_ID"
echo "Target: Results Directory: results_approx_N4_D2_*"
echo "=========================================="

# Run the script (ensure the filename matches exactly)
$PYTHON_PATH n4_approx.py

echo "=========================================="
echo "Job finished at $(date)"
echo "=========================================="