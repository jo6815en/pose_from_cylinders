#!/bin/bash -l

#SBATCH -A naiss2026-3-512-gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH -t 05:00:00
#SBATCH -J pose_train_test

#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -e

cd /nobackup/proj/disk/midlevel_representations/personal/johanna/pose_from_cylinders

module load GPU/Miniforge/26.3.2-2-eb

eval "$(conda shell.bash hook)"
conda activate pose_cylinders

echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "Python:"
which python
python --version

echo "Starting training..."

srun python train.py \
    --epochs 300 \
    --batch-size 16 \
    --output-dir runs/slurm_test

echo "Training finished."
