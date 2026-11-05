#!/bin/bash -l

#SBATCH -A naiss2026-3-512-gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH -t 24:00:00
#SBATCH -J vision_geometry
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Run mkdir -p logs before submitting: Slurm opens these files before execution.
# pipefail ensures training/evaluation failures are not hidden by tee.
set -eo pipefail

cd /nobackup/proj/disk/midlevel_representations/personal/johanna/pose_from_cylinders

# First argument: checkpoint from the occupancy run.
if [[ ! -f "$1" ]]; then
    echo "Occupancy checkpoint not found. Pass its path after the script name." >&2
    exit 1
fi

mkdir -p "runs/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}/evaluation"

module load GPU/Miniforge/26.3.2-2-eb

eval "$(conda shell.bash hook)"
conda activate pose_cylinders

echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Run directory: runs/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "Python:"
which python
python --version

echo "Starting geometry training..."
srun python -u train_vision.py \
    --stage geometry \
    --init-checkpoint "$1" \
    --data-dir /nobackup/proj/disk/midlevel_representations/personal/johanna/data \
    --lambda-depth-delta 0.35 \
    --epochs 200 \
    --batch-size 16 \
    --lr 5e-5 \
    --num-workers 8 \
    --seed 0 \
    --output-dir "runs/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}" \
    2>&1 | tee "runs/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}/train.log"

echo "Training finished. Evaluating best_vision.pt on testdataset..."
srun python -u eval_vision.py \
    --checkpoint "runs/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}/checkpoints/best_vision.pt" \
    --data-dir /nobackup/proj/disk/midlevel_representations/personal/johanna/data \
    --split testdataset \
    --batch-size 16 \
    --num-workers 8 \
    --occ-threshold 0.20 \
    2>&1 | tee "runs/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}/evaluation/test_best_vision.log"

echo "Evaluation finished: runs/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}/evaluation/test_best_vision.log"
