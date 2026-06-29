#!/bin/bash
#SBATCH --job-name=AME43_PAR
#SBATCH --output=ame43_par_%j.out
#SBATCH --error=ame43_par_err_%j.log
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

cd "$SLURM_SUBMIT_DIR"

# --- THREADING OPTIMIZATION ---
# CRITICAL: We have 8 parallel workers. 
# We limit each worker to 1 thread so they don't fight for cores.
# 8 workers * 1 thread = 8 cores used.
# The other 8 cores manage the GPU, Main Agent, and System Overhead.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1

# Python Environment
PYTHON_PATH="/home/sambhav/miniconda3/envs/quantum/bin/python"

echo "=========================================="
echo "Starting PARALLEL AME(4,3) on Bulbasaur..."
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPU(s): $SLURM_GPUS"
echo "Workdir: $SLURM_SUBMIT_DIR"
echo "=========================================="

# Run the Python script
$PYTHON_PATH ame43_parallel.py

echo "=========================================="
echo "Job finished at $(date)"
echo "=========================================="