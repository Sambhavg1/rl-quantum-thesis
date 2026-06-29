#!/bin/bash
#SBATCH --job-name=AME46_GPU2
#SBATCH --output=ame46_%j.out
#SBATCH --error=ame46_err_%j.log
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

cd $SLURM_SUBMIT_DIR

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTRAOP_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTEROP_THREADS=2

PYTHON_PATH="/home/sambhav/miniconda3/envs/quantum/bin/python"

echo "Starting AME(4,6) Search on Bulbasaur..."
echo "Date: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"

$PYTHON_PATH ame46_2.py

echo "Job Finished at $(date)"
