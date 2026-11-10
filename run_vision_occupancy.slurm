#!/bin/bash -l

#SBATCH -A naiss2026-3-512-gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH -t 4:00:00
#SBATCH -J vision_occupancy_big
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Run mkdir -p logs before submitting: Slurm opens these files before execution.
# pipefail ensures training/evaluation failures are not hidden by tee.
set -eo pipefail

cd /nobackup/proj/disk/midlevel_representations/personal/johanna/pose_from_cylinders

mkdir -p "runs/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}/evaluation"

module load GPU/Miniforge/26.9.4-1-eb
eval "$(conda shell.bash hook)"
conda activate /home/johanna/.conda-hpc/el9_gh200/envs/pose_cylinders

echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Run directory: runs/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "Python:"
which python
python --version

echo "Starting occupancy training..."
srun python -u train_vision.py \
    --stage occupancy \
    --epochs 200 \
    --batch-size 16 \
    --lr 5e-4 \
    --num-workers 4 \
    --seed 0 \
    --output-dir "runs/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}" \
    2>&1 | tee "runs/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}/train.log"

echo "Training finished. Evaluating best_vision.pt on testdataset..."
srun python -u eval_vision.py \
    --checkpoint "runs/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}/checkpoints/best_vision.pt" \
    --split testdataset \
    --batch-size 16 \
    --num-workers 8 \
    --occ-threshold 0.20 \
    2>&1 | tee "runs/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}/evaluation/test_best_vision.log"

echo "Evaluation finished: runs/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}/evaluation/test_best_vision.log"
