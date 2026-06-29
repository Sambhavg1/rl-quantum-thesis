#!/bin/bash
#SBATCH --job-name=AME46_CP
#SBATCH --output=ame46_cp_%j.out
#SBATCH --error=ame46_cp_err_%j.log
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

cd "$SLURM_SUBMIT_DIR"

# Allocate all 16 cores to matrix math and TF
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTRAOP_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTEROP_THREADS=2

PYTHON_PATH="/home/sambhav/miniconda3/envs/quantum/bin/python"

echo "=========================================="
echo "Starting AME(4,6) Ultimate CP-Gate Search..."
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "=========================================="

# Run the python script
$PYTHON_PATH ame46_cpgate.py

echo "=========================================="
echo "Job finished at $(date)"
echo "=========================================="