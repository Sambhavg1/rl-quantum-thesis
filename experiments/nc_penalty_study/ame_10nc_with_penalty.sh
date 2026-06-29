#!/bin/bash
#SBATCH --job-name=AME46_10A
#SBATCH --output=ame46_10a_%j.out
#SBATCH --error=ame46_10a_err_%j.log
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

cd "$SLURM_SUBMIT_DIR"

# Global GPU Memory fix for SLURM environments
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Allocate all 16 cores to matrix math and TF
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTRAOP_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTEROP_THREADS=2

PYTHON_PATH="/home/sambhav/miniconda3/envs/quantum/bin/python"

echo "=========================================="
echo "Starting Bulbasaur AME(4,6) 10-Angles Search..."
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "=========================================="

# Execute the script
$PYTHON_PATH ame46_10nc_with_penalty.py

echo "=========================================="
echo "Job finished at $(date)"
echo "=========================================="