#!/bin/bash -l

#SBATCH -A naiss2026-3-512-gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH -t 24:00:00
#SBATCH -J new_model

#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -e

cd /nobackup/proj/disk/midlevel_representations/personal/johanna/pose_from_cylinders

module load GPU/Miniforge/26.3.2-2-eb

eval "$(conda shell.bash hook)"
conda activate pose_cylinders

echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "Python:"
which python
python --version

echo "Starting training..."

srun python train.py \
    --epochs 300 \
    --batch-size 256 \
    --lr 5e-5\
    --num-workers 4\
    --output-dir runs/slurm_${SLURM_JOB_NAME}

echo "Training finished."
