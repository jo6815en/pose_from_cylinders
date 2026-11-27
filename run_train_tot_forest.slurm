#!/bin/bash -l

#SBATCH -A naiss2026-3-512-gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH -t 24:00:00
#SBATCH -J train_tot_forest
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Run mkdir -p logs before submitting: Slurm opens these files before execution.
# pipefail ensures a training failure is propagated through tee.
set -eo pipefail

cd /nobackup/proj/disk/midlevel_representations/personal/johanna/pose_from_cylinders

RUN_LOG="runs/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
mkdir -p "$(dirname "$RUN_LOG")"

module load GPU/Miniforge/26.9.4-1-eb
eval "$(conda shell.bash hook)"
conda activate /home/johanna/.conda-hpc/el9_gh200/envs/pose_cylinders

echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Training log: $RUN_LOG"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "Python:"
which python
python --version

echo "Starting forest-assisted total fine-tuning..."
srun python -u train_tot_forest.py 2>&1 | tee "$RUN_LOG"

echo "Training finished. Checkpoints and history are in runs/total_finetune_224_skrylle."
