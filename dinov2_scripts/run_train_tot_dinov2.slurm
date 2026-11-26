#!/bin/bash -l

#SBATCH -A naiss2026-3-512-gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH -t 24:00:00
#SBATCH -J train_tot_dinov2
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Create logs/ before submitting; Slurm opens these files before the job starts.
set -eo pipefail

cd /nobackup/proj/disk/midlevel_representations/personal/johanna/pose_from_cylinders

RUN_DIR="runs/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
VISION_JOB_ID="slurm_vision_occupancy_dinov2_5690832"
VISION_CHECKPOINT="runs/${VISION_JOB_ID}/checkpoints/best_vision.pt"
POSE_CHECKPOINT="runs/slurm_pose_only_new_2_5654299/checkpoints/best.pt"

mkdir -p "$RUN_DIR"

module load GPU/Miniforge/26.9.4-1-eb
eval "$(conda shell.bash hook)"
conda activate /home/johanna/.conda-hpc/el9_gh200/envs/pose_cylinders

echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Run directory: $RUN_DIR"
echo "DINOv2 vision checkpoint: $VISION_CHECKPOINT"
echo "Pose checkpoint: $POSE_CHECKPOINT"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "Python:"
which python
python --version

echo "Starting DINOv2 total fine-tuning..."
srun python -u train_tot_dinov2.py \
    --data-dir /nobackup/proj/disk/midlevel_representations/personal/johanna/data \
    --vision-checkpoint "$VISION_CHECKPOINT" \
    --pose-checkpoint "$POSE_CHECKPOINT" \
    --img-size 224 \
    --epochs 200 \
    --adapt-epochs 10 \
    --freeze-vision-epoch 90 \
    --pred-match-ramp-epochs 60 \
    --pred-match-max-prob 1.0 \
    --batch-size 16 \
    --lr 1e-5 \
    --pose-lr 1e-5 \
    --adapt-lr 1e-4 \
    --val-interval 10 \
    --num-workers 4 \
    --seed 0 \
    --output-dir "$RUN_DIR" \
    2>&1 | tee "$RUN_DIR/train.log"

echo "DINOv2 total fine-tuning finished. Checkpoints: $RUN_DIR/checkpoints/"
