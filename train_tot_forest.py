"""Standalone training script equivalent to the training portion of train_tot_forest.ipynb.

Run from the repository root after configuring the checkpoint and dataset paths below.
"""


import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"



import argparse
import json
import os
import random
import numpy as np
import time
import torch
from torch.utils.data import DataLoader
from dataset import SceneTwoPairsDataset
from forest_dataset import ForestPairDataset
from augmentations import build_train_transform, build_eval_transform
from model import PairImageCylinderModel
from metrics import relative_cylinder_errors
from losses import (
    vision_loss,
    patch_correspondence_loss,
    relative_depth_structure_loss,
    matched_depth_delta_loss,
    masked_relative_pose_loss_2d,
    matched_reprojection_loss_2d
)
from pathlib import Path
import matplotlib.pyplot as plt
from utils import plot_estimated_cylinders_on_images, plot_relative_pose, pose_to_text
from utils import move_batch_to_device


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_DIR = Path(".")
OUTPUT_DIR = Path("runs/total_finetune_224_skrylle")
#VISION_CHECKPOINT = Path("runs/vision_geometry_org_new_corr_new_data/checkpoints/best_vision.pt")
VISION_CHECKPOINT = Path("runs/vision_semi_good/checkpoints/latest.pt")
POSE_CHECKPOINT = Path("arrhenius_runs/slurm_pose_only_5842490/checkpoints/best.pt")

BATCH_SIZE = 8
NUM_EPOCHS = 200  # Total epochs; vision freezes after FREEZE_VISION_EPOCH.
ADAPT_EPOCHS = 10
FREEZE_VISION_EPOCH = 50  # Epochs 1–50 train vision; epoch 51 onward is pose-only.

PRED_MATCH_RAMP_EPOCHS = FREEZE_VISION_EPOCH - ADAPT_EPOCHS
PRED_MATCH_MAX_PROB = 1.0  # Reach 100% predicted matches by the freeze phase.

ADAPT_LEARNING_RATE = 1e-4
LEARNING_RATE = 1e-5  # Vision LR during joint fine-tuning.
POSE_LEARNING_RATE = 1e-5  # Pose LR during joint fine-tuning.

VAL_INTERVAL = 10
SEED = 0

NUM_WORKERS = 4
IMG_SIZE = 224
PATCH_SIZE = (16, 4)
NUM_BINS = 128
EMBED_DIM = 192
DEPTH = 4
NUM_HEADS = 4
DROPOUT = 0.1
 
LAMBDA_OCC = 10.0
LAMBDA_RADIUS = 10.0
OCC_THRESH = 0.5
LAMBDA_CORR = 1.0
LAMBDA_POSE = 2.0
LAMBDA_DEPTH_DELTA = 0.35
LAMBDA_WIDTH = 0.1
LAMBDA_SINKHORN = 0.1

BEST_MODEL_PATH = str(OUTPUT_DIR / "checkpoints" / "best.pt")
LATEST_MODEL_PATH = str(OUTPUT_DIR / "checkpoints" / "latest.pt")
RESUME_TRAINING = False
RESUME_CHECKPOINT_PATH = LATEST_MODEL_PATH

BEST_ADAPT_PATH = str(OUTPUT_DIR / "checkpoints" / "best_adapt.pt")
BEST_POSE_PATH = str(OUTPUT_DIR / "checkpoints" / "best_pose.pt")
INFERENCE_MODEL = "best_pose"  # "best", "best_pose", "latest", or "best_adapt"
INFERENCE_BATCH_SIZE = 4
INFERENCE_SHOW_TEXT = False
INFERENCE_SHOW_IMAGES = True
VISION_SAMPLE_IDX = 0
FOREST_ENABLED = True
FOREST_START_EPOCH = 1
FOREST_BATCH_SIZE = 2
FOREST_SAMPLE_IDX = 0
FOREST_OCC_THRESHOLD = 0.3

# Default keeps the existing training/evaluation data unchanged.
# Set FOREST_DATASET_PRESET to one of the Skrylle entries to use that stride
# for the forest batches used during training.
FOREST_DATASET_PRESET = "skrylle_stride_20"

FOREST_DATASET_PRESETS = {
    "torpa": (DATA_DIR / "forest_torpa", "relative_poses.json"),
    "skrylle_stride_1": (DATA_DIR / "forestdataset" / "skrylle", "relative_poses_stride_1.json"),
    "skrylle_stride_5": (DATA_DIR / "forestdataset" / "skrylle", "relative_poses_stride_5.json"),
    "skrylle_stride_10": (DATA_DIR / "forestdataset" / "skrylle", "relative_poses_stride_10.json"),
    "skrylle_stride_20": (DATA_DIR / "forestdataset" / "skrylle", "relative_poses_stride_20.json"),
}

# Evaluate these independently in the forest evaluation cell. To compare all
# Skrylle baselines, use all four skrylle_stride_* names here.
FOREST_EVAL_PRESETS = ["snoge_stride_*"]



# Check component settings, configure device and seeds (same as train_tot.py).
parser = argparse.ArgumentParser()
if not 0 <= ADAPT_EPOCHS < FREEZE_VISION_EPOCH < NUM_EPOCHS:
    parser.error("Require 0 <= --adapt-epochs < --freeze-vision-epoch < --epochs")

if min(LEARNING_RATE, ADAPT_LEARNING_RATE, LAMBDA_POSE) <= 0:
    parser.error("Learning rates and --lambda-pose must be positive")

POSE_LEARNING_RATE = LEARNING_RATE if POSE_LEARNING_RATE is None else POSE_LEARNING_RATE

if POSE_LEARNING_RATE <= 0 or min(VAL_INTERVAL, BATCH_SIZE) < 1:
    parser.error("Pose learning rate, intervals and batch size must be positive")

pose_saved = torch.load(POSE_CHECKPOINT, map_location="cpu", weights_only=False)

if "model_config" not in pose_saved:
    parser.error("--pose-checkpoint must be the standalone matched-pair pose checkpoint")

pose_cfg = pose_saved["model_config"]

POSE_EMBED_DIM = pose_cfg["embed_dim"]
POSE_NUM_HEADS = pose_cfg["num_heads"]
POSE_DEPTH = pose_cfg["depth"]
POSE_DROPOUT = pose_cfg["dropout"]
FOV_DEGREES = pose_cfg["fov_degrees"]
RADIUS_SCALE = pose_cfg["radius_scale"]
DISTANCE_SCALE = pose_cfg["distance_scale"]

vision_saved = torch.load(VISION_CHECKPOINT, map_location="cpu", weights_only=False)
vision_cfg = vision_saved.get("args", {})

for key in ("img_size", "patch_size", "embed_dim", "depth", "num_heads", "num_bins", "dropout"):
    if key in vision_cfg:
        saved_value, requested_value = (vision_cfg[key], globals()[key.upper()])
        if key == "patch_size":
            saved_value, requested_value = (tuple(saved_value), tuple(requested_value))
        if saved_value != requested_value:
            parser.error(
                f"Vision checkpoint {key}={saved_value}, but script uses {requested_value}; "
                f"set --{key.replace('_', '-')} to match"
            )

if abs(vision_cfg.get("fov_degrees", 90.0) - FOV_DEGREES) > 1e-06:
    parser.error("Vision and pose checkpoints must use the same field of view")

del vision_saved

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_float32_matmul_precision("high")

if device.type == "cuda":
    torch.backends.cudnn.benchmark = False

os.makedirs(OUTPUT_DIR, exist_ok=True)

os.makedirs(os.path.join(OUTPUT_DIR, "checkpoints"), exist_ok=True)

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

print("Seed: ", SEED)


# Fresh runs initialize from the vision and GT-pose checkpoints. For continuation, set RESUME_TRAINING=True.
print("Vision checkpoint:", VISION_CHECKPOINT)
print("GT-pose checkpoint:", POSE_CHECKPOINT)
print(f"Resume training: {RESUME_TRAINING}; checkpoint: {RESUME_CHECKPOINT_PATH}")
print(f"Pose adaptation through epoch {ADAPT_EPOCHS}; joint training through epoch {FREEZE_VISION_EPOCH}; pose-only refinement through epoch {NUM_EPOCHS}.")


train_dataset = SceneTwoPairsDataset(
    root_dir=os.path.join(DATA_DIR, "dataset"),
    image_size=IMG_SIZE,
    debug=False,
    return_two_pairs=False,
    transform=build_train_transform(IMG_SIZE),
)

val_dataset = SceneTwoPairsDataset(
    root_dir=os.path.join(DATA_DIR, "valdataset"),
    image_size=IMG_SIZE,
    debug=False,
    return_two_pairs=False,
    transform=build_eval_transform(IMG_SIZE),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=device.type == "cuda",
    persistent_workers=NUM_WORKERS > 0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=device.type == "cuda",
    persistent_workers=NUM_WORKERS > 0,
)

print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Train batches:", len(train_loader))
print("Validation batches:", len(val_loader))

model = PairImageCylinderModel(
    img_size=IMG_SIZE,
    patch_size=PATCH_SIZE,
    in_chans=3,
    embed_dim=EMBED_DIM,
    depth=DEPTH,
    num_heads=NUM_HEADS,
    num_bins=NUM_BINS,
    dropout=DROPOUT,
    pose_embed_dim=POSE_EMBED_DIM,
    pose_num_heads=POSE_NUM_HEADS,
    pose_depth=POSE_DEPTH,
    pose_dropout=POSE_DROPOUT,
    fov_degrees=FOV_DEGREES,
    radius_scale=RADIUS_SCALE,
    distance_scale=DISTANCE_SCALE,
).to(device)

model.load_vision_checkpoint(VISION_CHECKPOINT)

model.pose_head.load_state_dict(pose_saved["model_state_dict"], strict=True)

print("Loaded vision weights:", VISION_CHECKPOINT)
print("Loaded GT-pose weights:", POSE_CHECKPOINT)

del pose_saved

n_params = sum((p.numel() for p in model.parameters()))

print(f"Model parameters: {n_params / 1000000.0:.2f} M")

raw_model = model

vision_parameters = (
    list(model.backbone.parameters()) + list(model.vision_head.parameters()) + list(model.corr_decoder.parameters())
)

optimizer = torch.optim.AdamW(
    [
        {"params": vision_parameters, "lr": LEARNING_RATE, "name": "vision"},
        {"params": list(model.pose_head.parameters()), "lr": ADAPT_LEARNING_RATE, "name": "pose"},
    ],
    weight_decay=0.05,
)

#scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
scaler = torch.amp.GradScaler("cuda", enabled=False)

resume_checkpoint = None
START_EPOCH = 0
if RESUME_TRAINING:
    resume_path = Path(RESUME_CHECKPOINT_PATH)
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    resume_checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
    for key in ("model_state_dict", "optimizer_state_dict", "epoch", "history"):
        if key not in resume_checkpoint:
            raise ValueError(f"Checkpoint is missing required resume field: {key}")
    raw_model.load_state_dict(resume_checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
    if "scaler_state_dict" in resume_checkpoint:
        scaler.load_state_dict(resume_checkpoint["scaler_state_dict"])
    START_EPOCH = int(resume_checkpoint["epoch"])
    rng_state = resume_checkpoint.get("rng_state")
    if rng_state is not None:
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"].cpu())
        if device.type == "cuda" and rng_state.get("cuda") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in rng_state["cuda"]])
    if START_EPOCH >= NUM_EPOCHS:
        raise ValueError(f"Checkpoint is already at epoch {START_EPOCH}; NUM_EPOCHS is {NUM_EPOCHS}.")
    print(f"Resuming from epoch {START_EPOCH}; next epoch is {START_EPOCH + 1}.")
else:
    print("Starting from the supplied vision and GT-pose checkpoints.")

forest_train_loader = None
if FOREST_ENABLED:
    if FOREST_DATASET_PRESET not in FOREST_DATASET_PRESETS:
        raise ValueError(f"Unknown FOREST_DATASET_PRESET: {FOREST_DATASET_PRESET}")
    forest_train_dir, forest_train_poses = FOREST_DATASET_PRESETS[FOREST_DATASET_PRESET]
    forest_train_dataset = ForestPairDataset(
        dataset_dir=forest_train_dir,
        poses_file=forest_train_poses,
        transform=build_eval_transform(IMG_SIZE),
    )
    forest_train_loader = DataLoader(
        forest_train_dataset, batch_size=FOREST_BATCH_SIZE, shuffle=True, num_workers=0,
        pin_memory=device.type == "cuda", drop_last=True,
    )
    if len(forest_train_dataset) < FOREST_BATCH_SIZE:
        raise ValueError("Forest dataset must contain at least FOREST_BATCH_SIZE pairs.")
    print(
        f"Forest training data: {FOREST_DATASET_PRESET} "
        f"({forest_train_dir / forest_train_poses}); "
        f"samples: {len(forest_train_dataset)}; batch size: {FOREST_BATCH_SIZE}"
    )


def set_training_stage(model, optimizer, pose_only, freeze_vision=False):
    """Freeze vision for adaptation or final pose-only refinement."""
    freeze_vision_modules = pose_only or freeze_vision
    for module in (model.backbone, model.vision_head, model.corr_decoder):
        module.requires_grad_(not freeze_vision_modules)
    model.pose_head.requires_grad_(True)
    for group in optimizer.param_groups:
        if group["name"] == "vision":
            group["lr"] = 0.0 if freeze_vision_modules else LEARNING_RATE
        elif group["name"] == "pose":
            group["lr"] = ADAPT_LEARNING_RATE if pose_only else POSE_LEARNING_RATE


@torch.no_grad()
def validate(
    model,
    loader,
    device,
    occ_thresh,
    lambda_occ,
    lambda_radius,
    lambda_corr,
    lambda_depth_delta,
    lambda_pose,
    lambda_width,
    lambda_sinkhorn=0.0,
    pose_only=False,
    matching_mode="predicted",
):
    model.eval()
    totals = {
        "total": torch.zeros((), device=device),
        "vision": torch.zeros((), device=device),
        "vision_occupancy": torch.zeros((), device=device),
        "vision_radius": torch.zeros((), device=device),
        "vision_depth": torch.zeros((), device=device),
        "vision_width": torch.zeros((), device=device),
        "objective": torch.zeros((), device=device),
        "correspondence": torch.zeros((), device=device),
        "depth_structure": torch.zeros((), device=device),
        "depth_delta": torch.zeros((), device=device),
        "sinkhorn_reprojection": torch.zeros((), device=device),
        "pose": torch.zeros((), device=device),
        "translation": torch.zeros((), device=device),
        "rotation": torch.zeros((), device=device),
        "cylinder_radius_rel_l2": torch.zeros((), device=device),
        "cylinder_position_rel_l2": torch.zeros((), device=device),
        "pose_valid_samples": torch.zeros((), device=device),
        "pose_samples": torch.zeros((), device=device),
    }
    num_batches = 0
    total_sinkhorn_matches = torch.zeros((), device=device)
    for batch in loader:
        img_a, vision_a, img_b, vision_b, pose_ab = move_batch_to_device(batch, device)
        pred_vision_a, pred_vision_b, pred_pose, corr_a, corr_b, pose_info = model(
            img_a,
            img_b,
            return_corr=True,
            compute_pose=lambda_pose != 0,
            target_vision_a=vision_a,
            target_vision_b=vision_b,
            return_pose_inputs=True,
            occ_thresh=occ_thresh,
            matching_mode=matching_mode,
        )

        vis_a, occ_a, rad_a, dep_a, width_a = vision_loss(
            pred_vision_a,
            vision_a,
            occ_thresh=occ_thresh,
            lambda_occ=lambda_occ,
            lambda_radius=lambda_radius,
            lambda_depth=1.0,
            lambda_width=lambda_width,
        )
        vis_b, occ_b, rad_b, dep_b, width_b = vision_loss(
            pred_vision_b,
            vision_b,
            occ_thresh=occ_thresh,
            lambda_occ=lambda_occ,
            lambda_radius=lambda_radius,
            lambda_depth=1.0,
            lambda_width=lambda_width,
        )
        vision_loss_total = 0.5 * (vis_a + vis_b)

        corr_loss = patch_correspondence_loss(corr_a, vision_a, corr_b, vision_b)
        
        depth_structure_loss = 0.5 * (
            relative_depth_structure_loss(pred_vision_a, vision_a, occ_thresh)
            + relative_depth_structure_loss(pred_vision_b, vision_b, occ_thresh)
        )
        depth_delta_loss = matched_depth_delta_loss(pred_vision_a, vision_a, pred_vision_b, vision_b, occ_thresh)
        
        if lambda_pose != 0 and lambda_sinkhorn > 0:
            sinkhorn_reproj_loss, sinkhorn_match_count = matched_reprojection_loss_2d(
                pred_vision_a.float(),
                pred_vision_b=pred_vision_b.float(),
                relative_pose_pred=pred_pose.float(),
                occ_thresh=occ_thresh,
                fov_degrees=FOV_DEGREES,
                matching_mode="sinkhorn",
                return_match_count=True,
            )
            total_sinkhorn_matches += sinkhorn_match_count
        else:
            sinkhorn_reproj_loss = vision_loss_total.new_zeros(())

        if lambda_pose != 0:
            pose_total, t_loss, r_loss = masked_relative_pose_loss_2d(
                pred_pose, pose_ab, pose_info["pose_valid"], t_weight=1.0, r_weight=1.0
            )
        else:
            pose_total = vision_loss_total.new_zeros(())
            t_loss = vision_loss_total.new_zeros(())
            r_loss = vision_loss_total.new_zeros(())

        total_loss = (
            vision_loss_total
            + lambda_corr * corr_loss
            + 0.5 * depth_structure_loss
            + lambda_depth_delta * depth_delta_loss
            + lambda_pose * pose_total
            + lambda_sinkhorn * sinkhorn_reproj_loss
        )
        radius_a, position_a = relative_cylinder_errors(pred_vision_a, vision_a, occ_thresh)
        radius_b, position_b = relative_cylinder_errors(pred_vision_b, vision_b, occ_thresh)
        objective = lambda_pose * pose_total if pose_only else total_loss

        totals["objective"] += objective.detach()
        totals["total"] += total_loss.detach()
        totals["vision"] += vision_loss_total.detach()
        totals["vision_occupancy"] += (0.5 * (occ_a + occ_b)).detach()
        totals["vision_radius"] += (0.5 * (rad_a + rad_b)).detach()
        totals["vision_depth"] += (0.5 * (dep_a + dep_b)).detach()
        totals["vision_width"] += (0.5 * (width_a + width_b)).detach()
        totals["correspondence"] += corr_loss.detach()
        totals["depth_structure"] += depth_structure_loss.detach()
        totals["depth_delta"] += depth_delta_loss.detach()
        totals["sinkhorn_reprojection"] += sinkhorn_reproj_loss.detach()
        totals["pose"] += pose_total.detach()
        totals["pose_valid_samples"] += pose_info["pose_valid"].sum().detach()
        totals["pose_samples"] += pose_ab.shape[0]
        totals["translation"] += t_loss.detach()
        totals["rotation"] += r_loss.detach()
        totals["cylinder_radius_rel_l2"] += (0.5 * (radius_a + radius_b)).detach()
        totals["cylinder_position_rel_l2"] += (0.5 * (position_a + position_b)).detach()

        num_batches += 1

    if num_batches == 0:
        raise RuntimeError("Validation loader is empty.")
    metric_keys = list(totals)
    metric_values = torch.stack([totals[key] / num_batches for key in metric_keys] + [total_sinkhorn_matches]).cpu().tolist()
    metrics = dict(zip(metric_keys, metric_values))
    metrics["sinkhorn_matches_per_sample"] = metric_values[-1] / max(metrics["pose_samples"], 1.0)
    metrics["pose_valid_fraction"] = metrics["pose_valid_samples"] / max(metrics["pose_samples"], 1.0)
    return metrics


metric_names = ("vision_occupancy", "vision_radius", "vision_depth", "vision_width")
history = {
    key: []
    for key in (
        "epoch",
        "stage",
        "vision_lr",
        "pose_lr",
        "pred_match_probability",
        "train_pred_match_batch_fraction",
        "objective",
        "loss",
        "vision_loss",
        "correspondence_loss",
        "depth_structure_loss",
        "depth_delta_loss",
        "sinkhorn_reprojection_loss",
        "forest_sinkhorn_loss",
        "sinkhorn_matches_per_sample",
        "pose_loss",
        "translation_loss",
        "rotation_loss",
        "val_epoch",
        "val_stage",
        "val_objective",
        "val_total",
        "val_vision",
        "val_correspondence",
        "val_depth_structure",
        "val_depth_delta",
        "val_sinkhorn_reprojection",
        "val_sinkhorn_matches_per_sample",
        "val_pose",
        "val_gt_pose",
        "val_pred_pose",
        "val_gt_pose_valid_fraction",
        "val_pred_pose_valid_fraction",
        "val_translation",
        "val_rotation",
    )
}
weights = dict(
    vision_occupancy=LAMBDA_OCC, vision_radius=LAMBDA_RADIUS, vision_depth=1.0, vision_width=LAMBDA_WIDTH
)
for component in metric_names:
    for prefix in ("", "val_"):
        history[prefix + component] = []
        history[prefix + component + "_weighted"] = []

best_val_loss = float("inf")
best_adapt_pose = float("inf")
best_pose_loss = float("inf")

if resume_checkpoint is not None:
    history = resume_checkpoint["history"]
    required_history = (
        "epoch", "pred_match_probability", "train_pred_match_batch_fraction",
        "val_epoch", "val_gt_pose", "val_pred_pose",
        "val_gt_pose_valid_fraction", "val_pred_pose_valid_fraction",
    )
    missing_history = [key for key in required_history if key not in history]
    if missing_history:
        raise ValueError(
            "Resume checkpoint is incompatible with this notebook version; missing history keys: "
            + ", ".join(missing_history)
        )
    best_val_loss = resume_checkpoint.get("best_val_loss", float("inf"))
    best_adapt_pose = resume_checkpoint.get("best_adapt_pose", float("inf"))
    best_pose_loss = resume_checkpoint.get(
        "best_pose_loss", min(history["val_pred_pose"], default=float("inf"))
    )
    print(f"Restored history through epoch {START_EPOCH}.")

history.setdefault("forest_sinkhorn_loss", [float("nan")] * len(history.get("epoch", [])))
history.setdefault("sinkhorn_matches_per_sample", [float("nan")] * len(history.get("epoch", [])))
history.setdefault("val_sinkhorn_matches_per_sample", [float("nan")] * len(history.get("val_epoch", [])))

previous_stage = None
forest_iterator = None

for epoch in range(START_EPOCH + 1, NUM_EPOCHS + 1):
    epoch_start = time.perf_counter()
    pose_only = epoch <= ADAPT_EPOCHS
    freeze_vision = epoch > FREEZE_VISION_EPOCH
    stage = "pose_adapt" if pose_only else ("pose_refine" if freeze_vision else "joint")
    pose_objective_only = pose_only or freeze_vision
    joint_epochs_elapsed = max(0, epoch - ADAPT_EPOCHS)
    ramp_fraction = min(joint_epochs_elapsed / max(PRED_MATCH_RAMP_EPOCHS, 1), 1.0)
    pred_match_probability = (
        0.0 if pose_only else (1.0 if freeze_vision else PRED_MATCH_MAX_PROB * ramp_fraction)
    )
    set_training_stage(model, optimizer, pose_only, freeze_vision=freeze_vision)
    if stage != previous_stage:
        print(
            f"\nStage: {stage}; vision lr={optimizer.param_groups[0]['lr']:g}; "
            f"pose lr={optimizer.param_groups[1]['lr']:g}",
            flush=True,
        )
        previous_stage = stage
    model.train()
    if pose_objective_only:
        model.backbone.eval()
        model.vision_head.eval()
        model.corr_decoder.eval()
    pred_match_batches = 0
    if FOREST_ENABLED and epoch >= FOREST_START_EPOCH and forest_iterator is None:
        forest_iterator = iter(forest_train_loader)
    train_totals = {
        "total": torch.zeros((), device=device),
        "vision": torch.zeros((), device=device),
        "vision_occupancy": torch.zeros((), device=device),
        "vision_radius": torch.zeros((), device=device),
        "vision_depth": torch.zeros((), device=device),
        "vision_width": torch.zeros((), device=device),
        "objective": torch.zeros((), device=device),
        "correspondence": torch.zeros((), device=device),
        "depth_structure": torch.zeros((), device=device),
        "depth_delta": torch.zeros((), device=device),
        "sinkhorn_reprojection": torch.zeros((), device=device),
        "sinkhorn_match_count": torch.zeros((), device=device),
        "forest_sinkhorn": torch.zeros((), device=device),
        "pose": torch.zeros((), device=device),
        "translation": torch.zeros((), device=device),
        "rotation": torch.zeros((), device=device),
    }
    for batch in train_loader:
        matching_mode = "predicted" if random.random() < pred_match_probability else "gt"
        pred_match_batches += int(matching_mode == "predicted")
        img_a, vision_a, img_b, vision_b, pose_ab = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        forest_batch = None
        if FOREST_ENABLED and epoch >= FOREST_START_EPOCH:
            try:
                forest_batch = next(forest_iterator)
            except StopIteration:
                forest_iterator = iter(forest_train_loader)
                forest_batch = next(forest_iterator)
            img_a_forest, img_b_forest, _ = move_batch_to_device(forest_batch, device)

        #with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        with torch.autocast(device_type=device.type, enabled=False):
            model_outputs = model(
                img_a,
                img_b,
                return_corr=not pose_objective_only,
                compute_pose=LAMBDA_POSE != 0,
                target_vision_a=vision_a,
                target_vision_b=vision_b,
                return_pose_inputs=True,
                occ_thresh=OCC_THRESH,
                matching_mode=matching_mode,
            )
            if pose_objective_only:
                pred_vision_a, pred_vision_b, pred_pose, pose_info = model_outputs
                corr_a = corr_b = None
            else:
                pred_vision_a, pred_vision_b, pred_pose, corr_a, corr_b, pose_info = model_outputs
            if pose_objective_only:
                # These batches optimize pose only; skip the unused vision losses.
                if LAMBDA_POSE != 0:
                    pose_total, t_loss, r_loss = masked_relative_pose_loss_2d(
                        pred_pose, pose_ab, pose_info["pose_valid"], t_weight=1.0, r_weight=1.0
                    )
                else:
                    pose_total = pred_vision_a.new_zeros(())
                    t_loss = pred_vision_a.new_zeros(())
                    r_loss = pred_vision_a.new_zeros(())
                loss = LAMBDA_POSE * pose_total
            else:
                vis_a, occ_a, rad_a, dep_a, width_a = vision_loss(
                    pred_vision_a,
                    vision_a,
                    occ_thresh=OCC_THRESH,
                    lambda_occ=LAMBDA_OCC,
                    lambda_radius=LAMBDA_RADIUS,
                    lambda_depth=1.0,
                    lambda_width=LAMBDA_WIDTH,
                )
                vis_b, occ_b, rad_b, dep_b, width_b = vision_loss(
                    pred_vision_b,
                    vision_b,
                    occ_thresh=OCC_THRESH,
                    lambda_occ=LAMBDA_OCC,
                    lambda_radius=LAMBDA_RADIUS,
                    lambda_depth=1.0,
                    lambda_width=LAMBDA_WIDTH,
                )
                vision_loss_total = 0.5 * (vis_a + vis_b)

                corr_loss = patch_correspondence_loss(corr_a, vision_a, corr_b, vision_b)

                depth_structure_loss = 0.5 * (
                    relative_depth_structure_loss(pred_vision_a, vision_a, OCC_THRESH)
                    + relative_depth_structure_loss(pred_vision_b, vision_b, OCC_THRESH)
                )
                depth_delta_loss = matched_depth_delta_loss(
                    pred_vision_a, vision_a, pred_vision_b, vision_b, OCC_THRESH
                )
                
                if LAMBDA_POSE != 0:
                    pose_total, t_loss, r_loss = masked_relative_pose_loss_2d(
                        pred_pose, pose_ab, pose_info["pose_valid"], t_weight=1.0, r_weight=1.0
                    )
                else:
                    pose_total = pred_vision_a.new_zeros(())
                    t_loss = pred_vision_a.new_zeros(())
                    r_loss = pred_vision_a.new_zeros(())
                # Sinkhorn reprojection is applied to forest pairs below only.
                sinkhorn_reproj_loss = pred_vision_a.new_zeros(())
                sinkhorn_match_count = 0
                
                loss = (
                    vision_loss_total
                    + LAMBDA_CORR * corr_loss
                    + 0.5 * depth_structure_loss
                    + LAMBDA_DEPTH_DELTA * depth_delta_loss
                    + LAMBDA_POSE * pose_total
                    + LAMBDA_SINKHORN * sinkhorn_reproj_loss
                )
            forest_sinkhorn_loss = pred_vision_a.new_zeros(())
            if forest_batch is not None:
                forest_pred_a, forest_pred_b, forest_pose = model(
                    img_a_forest, img_b_forest, compute_pose=True,
                    matching_mode="predicted", occ_thresh=OCC_THRESH,
                )
                forest_sinkhorn_loss = matched_reprojection_loss_2d(
                    forest_pred_a.float(), pred_vision_b=forest_pred_b.float(),
                    relative_pose_pred=forest_pose.float(), occ_thresh=OCC_THRESH,
                    fov_degrees=FOV_DEGREES, matching_mode="sinkhorn",
                )
                loss = loss + LAMBDA_SINKHORN * forest_sinkhorn_loss
        objective = LAMBDA_POSE * pose_total if pose_objective_only else loss
        if forest_batch is not None and pose_objective_only:
            objective = objective + LAMBDA_SINKHORN * forest_sinkhorn_loss
        valid_pose = pose_info["pose_valid"] & (pose_ab[:, :2].norm(dim=-1) > 1e-06)
        
        if not pose_objective_only or valid_pose.any() or forest_batch is not None:
            scaler.scale(objective).backward()
            scaler.step(optimizer)
            scaler.update()
        
        train_totals["objective"] += objective.detach()
        if not pose_objective_only:
            train_totals["total"] += loss.detach()
            train_totals["vision"] += vision_loss_total.detach()
            train_totals["vision_occupancy"] += (0.5 * (occ_a + occ_b)).detach()
            train_totals["vision_radius"] += (0.5 * (rad_a + rad_b)).detach()
            train_totals["vision_depth"] += (0.5 * (dep_a + dep_b)).detach()
            train_totals["vision_width"] += (0.5 * (width_a + width_b)).detach()
            train_totals["correspondence"] += corr_loss.detach()
            train_totals["depth_structure"] += depth_structure_loss.detach()
            train_totals["depth_delta"] += depth_delta_loss.detach()
            train_totals["sinkhorn_reprojection"] += sinkhorn_reproj_loss.detach()
            train_totals["sinkhorn_match_count"] += sinkhorn_match_count
        train_totals["forest_sinkhorn"] += forest_sinkhorn_loss.detach()
        train_totals["pose"] += pose_total.detach()
        train_totals["translation"] += t_loss.detach()
        train_totals["rotation"] += r_loss.detach()
    
    metric_keys = list(train_totals)
    metric_values = torch.stack([train_totals[key] / len(train_loader) for key in metric_keys]).cpu().tolist()
    train_metrics = dict(zip(metric_keys, metric_values))
    sinkhorn_match_count = train_metrics.pop("sinkhorn_match_count")
    train_metrics["sinkhorn_matches_per_sample"] = (
        sinkhorn_match_count / max(len(train_loader.dataset), 1)
    )
    
    if pose_objective_only:
        for key in ('total', 'vision', 'vision_occupancy', 'vision_radius', 'vision_depth', 'vision_width', 'correspondence', 'depth_structure', 'depth_delta', 'sinkhorn_reprojection', 'sinkhorn_matches_per_sample'):
            train_metrics[key] = float("nan")
    history["epoch"].append(epoch)
    history["stage"].append(stage)
    history["vision_lr"].append(optimizer.param_groups[0]["lr"])
    history["pose_lr"].append(optimizer.param_groups[1]["lr"])
    history["pred_match_probability"].append(pred_match_probability)
    history["train_pred_match_batch_fraction"].append(pred_match_batches / max(len(train_loader), 1))
    train_mapping = {
        "objective": "objective",
        "loss": "total",
        "vision_loss": "vision",
        "correspondence_loss": "correspondence",
        "depth_structure_loss": "depth_structure",
        "depth_delta_loss": "depth_delta",
        "sinkhorn_reprojection_loss": "sinkhorn_reprojection",
        "forest_sinkhorn_loss": "forest_sinkhorn",
        "sinkhorn_matches_per_sample": "sinkhorn_matches_per_sample",
        "pose_loss": "pose",
        "translation_loss": "translation",
        "rotation_loss": "rotation",
    }
    for key, metric in train_mapping.items():
        history[key].append(train_metrics[metric])
    for component, weight in weights.items():
        history[component].append(train_metrics[component])
        history[component + "_weighted"].append(weight * train_metrics[component])
    
    train_loss_log = (
        f" | tot={train_metrics['total']:.4f} | vis={train_metrics['vision']:.4f} "
        f"| occ={train_metrics['vision_occupancy']:.4f} | rad={train_metrics['vision_radius']:.4f} "
        f"| depth={train_metrics['vision_depth']:.4f} | width={train_metrics['vision_width']:.4f} "
        f"| corr={train_metrics['correspondence']:.4f} | struct={train_metrics['depth_structure']:.4f} "
        f"| delta={train_metrics['depth_delta']:.4f}"
        if not pose_objective_only else ""
    )
    log = (
        f"Epoch {epoch} [{stage}]: obj={train_metrics['objective']:.4f}"
        f"{train_loss_log} | pose={train_metrics['pose']:.4f} "
        f"| pred-match p={pred_match_probability:.2f} "
        f"(actual {pred_match_batches / max(len(train_loader), 1):.2f})"
        f" | forest_sinkhorn={train_metrics['forest_sinkhorn']:.4f}"
    )

    improved = improved_adapt = improved_pose = False
    should_validate = (
        epoch == 1 or epoch % VAL_INTERVAL == 0 or epoch in (ADAPT_EPOCHS, ADAPT_EPOCHS + 1, FREEZE_VISION_EPOCH, FREEZE_VISION_EPOCH + 1, NUM_EPOCHS)
    )
    if should_validate:
        val_pred_metrics = validate(
            model, val_loader, device, OCC_THRESH, LAMBDA_OCC,
            LAMBDA_RADIUS, LAMBDA_CORR, LAMBDA_DEPTH_DELTA,
            LAMBDA_POSE, LAMBDA_WIDTH, pose_only=pose_objective_only, matching_mode="predicted",
            lambda_sinkhorn=0.0,
        )
        # GT matching is diagnostic; run it less often than end-to-end validation.
        run_gt_validation = epoch == 1 or epoch % 50 == 0
        val_gt_metrics = None
        if run_gt_validation:
            val_gt_metrics = validate(
                model, val_loader, device, OCC_THRESH, LAMBDA_OCC,
                LAMBDA_RADIUS, LAMBDA_CORR, LAMBDA_DEPTH_DELTA,
                LAMBDA_POSE, LAMBDA_WIDTH, pose_only=pose_objective_only, matching_mode="gt",
                lambda_sinkhorn=0.0,
            )
        # The primary val_* series and best_pose checkpoint track predicted matching.
        val_metrics = val_pred_metrics
        history["val_epoch"].append(epoch)
        history["val_stage"].append(stage)
        history["val_gt_pose"].append(val_gt_metrics["pose"] if val_gt_metrics is not None else float("nan"))
        history["val_pred_pose"].append(val_pred_metrics["pose"])
        history["val_gt_pose_valid_fraction"].append(
            val_gt_metrics["pose_valid_fraction"] if val_gt_metrics is not None else float("nan")
        )
        history["val_pred_pose_valid_fraction"].append(val_pred_metrics["pose_valid_fraction"])
        for key, metric in {
            "val_objective": "objective",
            "val_total": "total",
            "val_vision": "vision",
            "val_correspondence": "correspondence",
            "val_depth_structure": "depth_structure",
            "val_depth_delta": "depth_delta",
            "val_sinkhorn_reprojection": "sinkhorn_reprojection",
            "val_sinkhorn_matches_per_sample": "sinkhorn_matches_per_sample",
            "val_pose": "pose",
            "val_translation": "translation",
            "val_rotation": "rotation",
        }.items():
            history[key].append(val_metrics[metric])
        for component, weight in weights.items():
            history["val_" + component].append(val_metrics[component])
            history["val_" + component + "_weighted"].append(weight * val_metrics[component])
        gt_log = (
            f" | val_pose_GTmatch={val_gt_metrics['pose']:.4f} "
            f"(valid {val_gt_metrics['pose_valid_fraction']:.2f})"
            if val_gt_metrics is not None else ""
        )
        log += (
            f" | val_tot={val_metrics['total']:.4f} | val_vis={val_metrics['vision']:.4f} "
            f"| val_occ={val_metrics['vision_occupancy']:.4f} | val_rad={val_metrics['vision_radius']:.4f} "
            f"| val_depth={val_metrics['vision_depth']:.4f} | val_width={val_metrics['vision_width']:.4f} "
            f"{gt_log} "
            f"| val_pose_Predmatch={val_pred_metrics['pose']:.4f} "
            f"(valid {val_pred_metrics['pose_valid_fraction']:.2f})"
        )
        improved_pose = val_metrics["pose"] < best_pose_loss
        best_pose_loss = min(best_pose_loss, val_metrics["pose"])
        if pose_only:
            improved_adapt = val_metrics["pose"] < best_adapt_pose
            best_adapt_pose = min(best_adapt_pose, val_metrics["pose"])
        elif stage == "joint":
            improved = val_metrics["total"] < best_val_loss
            best_val_loss = min(best_val_loss, val_metrics["total"])
        if improved_pose:
            log += " | BEST POSE"
        if improved or improved_adapt:
            log += " | BEST ADAPT" if pose_only else " | BEST JOINT"
    state = {
        "epoch": epoch,
        "training_stage": stage,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "args": {
            "data_dir": str(DATA_DIR), "output_dir": str(OUTPUT_DIR),
            "vision_checkpoint": str(VISION_CHECKPOINT), "pose_checkpoint": str(POSE_CHECKPOINT),
            "epochs": NUM_EPOCHS, "adapt_epochs": ADAPT_EPOCHS,
            "freeze_vision_epoch": FREEZE_VISION_EPOCH,
            "pred_match_ramp_epochs": PRED_MATCH_RAMP_EPOCHS,
            "pred_match_max_prob": PRED_MATCH_MAX_PROB,
            "adapt_lr": ADAPT_LEARNING_RATE, "lr": LEARNING_RATE,
            "pose_lr": POSE_LEARNING_RATE, "batch_size": BATCH_SIZE,
            "val_interval": VAL_INTERVAL, "seed": SEED,
            "num_workers": NUM_WORKERS, "img_size": IMG_SIZE,
            "patch_size": PATCH_SIZE, "embed_dim": EMBED_DIM,
            "depth": DEPTH, "num_heads": NUM_HEADS, "num_bins": NUM_BINS,
            "dropout": DROPOUT, "lambda_occ": LAMBDA_OCC,
            "lambda_radius": LAMBDA_RADIUS, "occ_thresh": OCC_THRESH,
            "lambda_corr": LAMBDA_CORR, "lambda_depth_delta": LAMBDA_DEPTH_DELTA,
            "lambda_pose": LAMBDA_POSE, "lambda_width": LAMBDA_WIDTH,
            "pose_embed_dim": POSE_EMBED_DIM, "pose_num_heads": POSE_NUM_HEADS,
            "pose_depth": POSE_DEPTH, "pose_dropout": POSE_DROPOUT,
            "fov_degrees": FOV_DEGREES, "radius_scale": RADIUS_SCALE,
            "distance_scale": DISTANCE_SCALE,
        },
        "history": history,
        "best_val_loss": best_val_loss,
        "best_adapt_pose": best_adapt_pose,
        "best_pose_loss": best_pose_loss,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        },
    }
    torch.save(state, os.path.join(OUTPUT_DIR, "checkpoints", "latest.pt"))
    if improved:
        torch.save(state, os.path.join(OUTPUT_DIR, "checkpoints", "best.pt"))
    if improved_adapt:
        torch.save(state, os.path.join(OUTPUT_DIR, "checkpoints", "best_adapt.pt"))
    if improved_pose:
        torch.save(state, BEST_POSE_PATH)
    with open(os.path.join(OUTPUT_DIR, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(log + f" | time={time.perf_counter() - epoch_start:.1f}s", flush=True)

# Keep final weights available to the notebook's inference cells.
latest_model_state_dict = {key: value.detach().cpu().clone() for key, value in raw_model.state_dict().items()}
