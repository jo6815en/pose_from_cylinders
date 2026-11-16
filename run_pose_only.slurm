#!/bin/bash -l

#SBATCH -A naiss2026-3-512-gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH -t 4:00:00
#SBATCH -J pose_only
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Run mkdir -p logs before submitting: Slurm opens these files before execution.
# pipefail ensures training/evaluation failures are not hidden by tee.
set -eo pipefail

cd /nobackup/proj/disk/midlevel_representations/personal/johanna/pose_from_cylinders

RUN_DIR="runs/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
EVAL_DIR="$RUN_DIR/evaluation"
mkdir -p "$EVAL_DIR"

module load GPU/Miniforge/26.9.4-1-eb
eval "$(conda shell.bash hook)"
conda activate /home/johanna/.conda-hpc/el9_gh200/envs/pose_cylinders

echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Run directory: $RUN_DIR"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "Python:"
which python
python --version

echo "Starting pose-only training..."
srun python -u train_pose_only.py \
    --data-dir /nobackup/proj/disk/midlevel_representations/personal/johanna/data \
    --epochs 100 \
    --batch-size 16 \
    --lr 1e-4 \
    --val-interval 5 \
    --seed 0 \
    --output-dir "$RUN_DIR" \
    2>&1 | tee "$RUN_DIR/train.log"

echo "Training finished. Evaluating best.pt on testdataset..."
srun python -u train_pose_only.py \
    --data-dir /nobackup/proj/disk/midlevel_representations/personal/johanna/data \
    --eval-only \
    --checkpoint "$RUN_DIR/checkpoints/best.pt" \
    --split testdataset \
    --batch-size 16 \
    --output-dir "$EVAL_DIR" \
    2>&1 | tee "$EVAL_DIR/test_best_pose.log"

echo "Evaluation finished: $EVAL_DIR/test_best_pose.log"
