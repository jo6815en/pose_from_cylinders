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
# pipefail ensures training/evaluation failures are not hidden by tee.
set -eo pipefail

cd /nobackup/proj/disk/midlevel_representations/personal/johanna/pose_from_cylinders

RUN_DIR="runs/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR/evaluation"

module load GPU/Miniforge/26.9.4-1-eb
eval "$(conda shell.bash hook)"
conda activate /home/johanna/.conda-hpc/el9_gh200/envs/pose_cylinders

# Override these when submitting if you want to use different input checkpoints.
DATA_DIR="${DATA_DIR:-/nobackup/proj/disk/midlevel_representations/personal/johanna/data}"
VISION_CHECKPOINT="${VISION_CHECKPOINT:-runs/vision_semi_good/checkpoints/latest.pt}"
POSE_CHECKPOINT="${POSE_CHECKPOINT:-arrhenius_runs/slurm_pose_only_5842490/checkpoints/best.pt}"

echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Run directory: $RUN_DIR"
echo "Vision checkpoint: $VISION_CHECKPOINT"
echo "Pose checkpoint: $POSE_CHECKPOINT"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "Python:"
which python
python --version

echo "Starting forest-assisted total fine-tuning..."
srun python -u train_tot_forest.py \
    --data-dir "$DATA_DIR" \
    --vision-checkpoint "$VISION_CHECKPOINT" \
    --pose-checkpoint "$POSE_CHECKPOINT" \
    --epochs 200 \
    --adapt-epochs 10 \
    --freeze-vision-epoch 50 \
    --pred-match-ramp-epochs 40 \
    --pred-match-max-prob 1.0 \
    --batch-size 8 \
    --lr 1e-5 \
    --pose-lr 1e-5 \
    --adapt-lr 1e-4 \
    --val-interval 10 \
    --num-workers 4 \
    --seed 0 \
    --forest-dataset-preset skrylle_stride_20 \
    --forest-batch-size 2 \
    --forest-start-epoch 1 \
    --lambda-sinkhorn 0.1 \
    --output-dir "$RUN_DIR" \
    2>&1 | tee "$RUN_DIR/train.log"

echo "Training finished. Evaluating best_pose.pt on testdataset..."
srun python -u eval_pose.py \
    --checkpoint "$RUN_DIR/checkpoints/best_pose.pt" \
    --data-dir "$DATA_DIR" \
    --split testdataset \
    --batch-size 8 \
    --num-workers 4 \
    2>&1 | tee "$RUN_DIR/evaluation/test_best_pose.log"

echo "Evaluation finished: $RUN_DIR/evaluation/test_best_pose.log"
