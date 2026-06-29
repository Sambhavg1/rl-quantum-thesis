#!/bin/bash
#SBATCH --job-name=AME62_CONV
#SBATCH --output=ame62_conv_%j.out
#SBATCH --error=ame62_conv_err_%j.log
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

cd "$SLURM_SUBMIT_DIR"

# Allocate all 16 cores to matrix math and TensorFlow
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTRAOP_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTEROP_THREADS=2

# Environment Path
PYTHON_PATH="/home/sambhav/miniconda3/envs/quantum/bin/python"

echo "=========================================="
echo "Starting Action-Masked AME(6,2) Run..."
echo "Job ID: $SLURM_JOB_ID"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Batch Size: 16384 steps"
echo "=========================================="

# Run the python script
$PYTHON_PATH ame62_converge.py

echo "=========================================="
echo "Job finished at $(date)"
echo "=========================================="