"""Two-stage fine-tuning of the existing joint model.

--epochs includes the adaptation phase. Example: --epochs 100 --adapt-epochs 10
means 10 epochs pose-only adaptation, followed by 90 jointly trained epochs.
Vision and standalone GT-pose checkpoints are loaded once at startup. One
optimizer preserves pose momentum at the switch. GT bins and IDs still choose
pairs; the pose head always receives predicted radius/depth from RGB.

During adaptation: vision modules are frozen AND in eval mode; only the pose
loss is optimized. Full joint loss and vision components are logged as diagnostics.
During joint tuning: all modules train with the original weighted objective.
No extra pose-depth geometry consistency term is enabled.
"""

import argparse
import json
import os
import random
import numpy as np
import time

import torch
from torch.utils.data import DataLoader

from dataset import SceneTwoPairsDataset
from augmentations import build_train_transform, build_eval_transform
from model import PairImageCylinderModel
from metrics import relative_cylinder_errors
from losses import (
    vision_loss,
    patch_correspondence_loss,
    relative_depth_structure_loss,
    matched_depth_delta_loss,
    masked_relative_pose_loss_2d
)


def move_batch_to_device(batch, device):
    return tuple(x.to(device, non_blocking=True) for x in batch)


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    occ_thresh,
    lambda_occ,
    lambda_radius,
    lambda_corr,
    lambda_depth_delta,
    lambda_pose,
    lambda_width,
    scaler,
    pose_only=False,
):
    model.train()
    if pose_only:
        # requires_grad=False alone does not disable dropout/normalization state.
        model.backbone.eval()
        model.vision_head.eval()
        model.corr_decoder.eval()

    totals = {
        "total": 0.0,
        "vision": 0.0,
        "vision_occupancy": 0.0,
        "vision_radius": 0.0,
        "vision_depth": 0.0,
        "vision_width": 0.0,
        "objective": 0.0,
        "correspondence": 0.0,
        "depth_structure": 0.0,
        "depth_delta": 0.0,
        "pose": 0.0,
        "translation": 0.0,
        "rotation": 0.0,
    }

    for batch in loader:
        (
            img_a,
            vision_a,
            img_b,
            vision_b,
            pose_ab,
        ) = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            pred_vision_a, pred_vision_b, pred_pose, corr_a, corr_b, pose_info = model(
                    img_a, img_b, return_corr=True, compute_pose=(lambda_pose != 0),
                    target_vision_a=vision_a, target_vision_b=vision_b, return_pose_inputs=True, occ_thresh=occ_thresh)

            vis_a, occ_a, rad_a, dep_a, width_a = vision_loss(pred_vision_a, vision_a, occ_thresh=occ_thresh, lambda_occ=lambda_occ,
                lambda_radius=lambda_radius, lambda_depth=1.0, lambda_width=lambda_width)

            vis_b, occ_b, rad_b, dep_b, width_b = vision_loss(pred_vision_b, vision_b, occ_thresh=occ_thresh, lambda_occ=lambda_occ,
                lambda_radius=lambda_radius, lambda_depth=1.0, lambda_width=lambda_width)

            vision_loss_total = 0.5 * ( vis_a + vis_b)

            corr_loss = patch_correspondence_loss(corr_a, vision_a, corr_b, vision_b)

            depth_structure_loss = 0.5 * (relative_depth_structure_loss(pred_vision_a, vision_a,  occ_thresh)
                + relative_depth_structure_loss(pred_vision_b, vision_b, occ_thresh))

            depth_delta_loss = matched_depth_delta_loss(pred_vision_a, vision_a, pred_vision_b, vision_b, occ_thresh)

            if lambda_pose != 0:
                pose_total, t_loss, r_loss = masked_relative_pose_loss_2d(
                    pred_pose,
                    pose_ab,
                    pose_info["pose_valid"],
                    t_weight=1.0,
                    r_weight=1.0,
                )
            else:
                pose_total = vision_loss_total.new_zeros(())
                t_loss = vision_loss_total.new_zeros(())
                r_loss = vision_loss_total.new_zeros(())

            #pose_total, t_loss, r_loss = pose_loss(pred_pose, pose_ab)

            loss = (
                vision_loss_total
                + lambda_corr * corr_loss
                + 0.5 * depth_structure_loss
                + lambda_depth_delta * depth_delta_loss
                + lambda_pose * pose_total
            )

        # Log the full joint loss for comparison across both phases, but
        # optimize pose alone while adapting to frozen predicted geometry.
        objective = lambda_pose * pose_total if pose_only else loss
        valid_pose = pose_info["pose_valid"] & (pose_ab[:, :2].norm(dim=-1) > 1e-6)
        if not pose_only or valid_pose.any():
            scaler.scale(objective).backward()
            scaler.step(optimizer)
            scaler.update()

        totals["objective"] += objective.item()

        totals["total"] += loss.item()
        totals["vision"] += vision_loss_total.item()
        totals["vision_occupancy"] += (0.5 * (occ_a + occ_b)).item()
        totals["vision_radius"] += (0.5 * (rad_a + rad_b)).item()
        totals["vision_depth"] += (0.5 * (dep_a + dep_b)).item()
        totals["vision_width"] += (0.5 * (width_a + width_b)).item()
        totals["correspondence"] += corr_loss.item()
        totals["depth_structure"] += depth_structure_loss.item()
        totals["depth_delta"] += depth_delta_loss.item()
        totals["pose"] += pose_total.item()
        totals["translation"] += t_loss.item()
        totals["rotation"] += r_loss.item()

    return {
        key: value / len(loader)
        for key, value in totals.items()
    }

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
    pose_only=False,
):
    model.eval()

    totals = {
        "total": 0.0,
        "vision": 0.0,
        "vision_occupancy": 0.0,
        "vision_radius": 0.0,
        "vision_depth": 0.0,
        "vision_width": 0.0,
        "objective": 0.0,
        "correspondence": 0.0,
        "depth_structure": 0.0,
        "depth_delta": 0.0,
        "pose": 0.0,
        "translation": 0.0,
        "rotation": 0.0,
        "cylinder_radius_rel_l2": 0.0,
        "cylinder_position_rel_l2": 0.0,
    }

    num_batches = 0

    for batch in loader:
        (
            img_a,
            vision_a,
            img_b,
            vision_b,
            pose_ab,
        ) = move_batch_to_device(batch, device)

        pred_vision_a, pred_vision_b, pred_pose, corr_a, corr_b, pose_info = model(
                img_a, img_b, return_corr=True, compute_pose=(lambda_pose != 0),
                target_vision_a=vision_a, target_vision_b=vision_b, return_pose_inputs=True, occ_thresh=occ_thresh)

        vis_a, occ_a, rad_a, dep_a, width_a = vision_loss(pred_vision_a, vision_a, occ_thresh=occ_thresh, lambda_occ=lambda_occ,
            lambda_radius=lambda_radius, lambda_depth=1.0, lambda_width=lambda_width)
        vis_b, occ_b, rad_b, dep_b, width_b = vision_loss(pred_vision_b, vision_b, occ_thresh=occ_thresh, lambda_occ=lambda_occ,
            lambda_radius=lambda_radius, lambda_depth=1.0, lambda_width=lambda_width)
        vision_loss_total = 0.5 * (vis_a + vis_b)

        corr_loss = patch_correspondence_loss(corr_a, vision_a, corr_b, vision_b)

        depth_structure_loss = 0.5 * (
            relative_depth_structure_loss(pred_vision_a, vision_a, occ_thresh)
            + relative_depth_structure_loss(pred_vision_b, vision_b, occ_thresh)
        )

        depth_delta_loss = matched_depth_delta_loss(pred_vision_a, vision_a, pred_vision_b, vision_b, occ_thresh)

        # pose_total, t_loss, r_loss = pose_loss(pred_pose, pose_ab)

        if lambda_pose != 0:
            pose_total, t_loss, r_loss = masked_relative_pose_loss_2d(
                pred_pose,
                pose_ab,
                pose_info["pose_valid"],
                t_weight=1.0,
                r_weight=1.0,
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
        )

        radius_a, position_a = relative_cylinder_errors(
            pred_vision_a, vision_a, occ_thresh
        )
        radius_b, position_b = relative_cylinder_errors(
            pred_vision_b, vision_b, occ_thresh
        )

        objective = lambda_pose * pose_total if pose_only else total_loss
        totals["objective"] += objective.item()
        totals["total"] += total_loss.item()
        totals["vision"] += vision_loss_total.item()
        totals["vision_occupancy"] += (0.5 * (occ_a + occ_b)).item()
        totals["vision_radius"] += (0.5 * (rad_a + rad_b)).item()
        totals["vision_depth"] += (0.5 * (dep_a + dep_b)).item()
        totals["vision_width"] += (0.5 * (width_a + width_b)).item()
        totals["correspondence"] += corr_loss.item()
        totals["depth_structure"] += depth_structure_loss.item()
        totals["depth_delta"] += depth_delta_loss.item()
        totals["pose"] += pose_total.item()
        totals["translation"] += t_loss.item()
        totals["rotation"] += r_loss.item()
        totals["cylinder_radius_rel_l2"] += (
            0.5 * (radius_a + radius_b)
        ).item()
        totals["cylinder_position_rel_l2"] += (
            0.5 * (position_a + position_b)
        ).item()
        num_batches += 1

    if num_batches == 0:
        raise RuntimeError("Validation loader is empty.")

    metrics = {
        key: value / num_batches
        for key, value in totals.items()
    }

    return metrics


def set_training_stage(model, optimizer, pose_only, args):
    """Freeze vision weights for adaptation; unfreeze all for joint fine-tuning."""
    for module in (model.backbone, model.vision_head, model.corr_decoder):
        module.requires_grad_(not pose_only)
    model.pose_head.requires_grad_(True)
    for group in optimizer.param_groups:
        if group["name"] == "vision":
            group["lr"] = 0.0 if pose_only else args.lr
        elif group["name"] == "pose":
            group["lr"] = args.adapt_lr if pose_only else args.pose_lr


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        default="/nobackup/proj/disk/midlevel_representations/personal/johanna/data",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Vision learning rate during joint fine-tuning")
    parser.add_argument("--pose-lr", type=float, default=None,
                        help="Pose learning rate during joint fine-tuning; defaults to --lr")
    parser.add_argument("--adapt-lr", type=float, default=1e-4)
    parser.add_argument("--adapt-epochs", type=int, default=10,
                        help="First N of --epochs: adapt pose while vision is frozen")
    parser.add_argument("--vision-checkpoint", required=True)
    parser.add_argument("--pose-checkpoint", required=True)
    parser.add_argument("--val-interval", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--patch-size", type=int, nargs=2, default=(16, 4))
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-bins", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--occ-thresh", type=float, default=0.5)
    
    parser.add_argument("--lambda-occ", type=float, default=10.0)
    parser.add_argument("--lambda-radius", type=float, default=10.0)
    parser.add_argument("--lambda-width", type=float, default=0.1)
    parser.add_argument("--lambda-corr", type=float, default=1.0)
    parser.add_argument("--lambda-depth-delta", type=float, default=0.35)
    parser.add_argument("--lambda-pose", type=float, default=2.0)

    parser.add_argument("--output-dir", default="runs/total_finetune")

    args = parser.parse_args()
    if not 0 <= args.adapt_epochs < args.epochs:
        parser.error("Require 0 <= --adapt-epochs < --epochs (total epoch count)")
    if min(args.lr, args.adapt_lr, args.lambda_pose) <= 0:
        parser.error("Learning rates and --lambda-pose must be positive")
    args.pose_lr = args.lr if args.pose_lr is None else args.pose_lr
    if args.pose_lr <= 0 or min(args.val_interval, args.batch_size) < 1:
        parser.error("Pose learning rate, intervals and batch size must be positive")
    pose_saved = torch.load(args.pose_checkpoint, map_location="cpu", weights_only=False)
    if "model_config" not in pose_saved:
        parser.error("--pose-checkpoint must be the standalone matched-pair pose checkpoint")
    pose_cfg = pose_saved["model_config"]
    args.pose_embed_dim = pose_cfg["embed_dim"]
    args.pose_num_heads = pose_cfg["num_heads"]
    args.pose_depth = pose_cfg["depth"]
    args.pose_dropout = pose_cfg["dropout"]
    args.fov_degrees = pose_cfg["fov_degrees"]
    args.radius_scale = pose_cfg["radius_scale"]
    args.distance_scale = pose_cfg["distance_scale"]
    vision_saved = torch.load(args.vision_checkpoint, map_location="cpu", weights_only=False)
    vision_cfg = vision_saved.get("args", {})
    for key in ("img_size", "patch_size", "embed_dim", "depth", "num_heads", "num_bins", "dropout"):
        if key in vision_cfg:
            saved_value, requested_value = vision_cfg[key], getattr(args, key)
            if key == "patch_size":
                saved_value, requested_value = tuple(saved_value), tuple(requested_value)
            if saved_value != requested_value:
                parser.error(f"Vision checkpoint {key}={saved_value}, but script uses {requested_value}; set --{key.replace('_', '-')} to match")
    if abs(vision_cfg.get("fov_degrees", 90.) - args.fov_degrees) > 1e-6:
        parser.error("Vision and pose checkpoints must use the same field of view")
    del vision_saved

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA:", torch.version.cuda)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print("Seed: ", args.seed)

    train_dataset = SceneTwoPairsDataset(
        root_dir=os.path.join(args.data_dir, "dataset"),
        image_size=args.img_size,
        debug=False,
        return_two_pairs=False,
        transform=build_train_transform(args.img_size),
    )

    val_dataset = SceneTwoPairsDataset(
        root_dir=os.path.join(args.data_dir, "valdataset"),
        image_size=args.img_size,
        debug=False,
        return_two_pairs=False,
        transform=build_eval_transform(args.img_size),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    print("Train samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Train batches:", len(train_loader))
    print("Validation batches:", len(val_loader))

    model = PairImageCylinderModel(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=3,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        num_bins=args.num_bins,
        dropout=args.dropout,
        pose_embed_dim=args.pose_embed_dim, pose_num_heads=args.pose_num_heads,
        pose_depth=args.pose_depth, pose_dropout=args.pose_dropout,
        fov_degrees=args.fov_degrees, radius_scale=args.radius_scale,
        distance_scale=args.distance_scale,
    ).to(device)

    model.load_vision_checkpoint(args.vision_checkpoint)
    model.pose_head.load_state_dict(pose_saved["model_state_dict"], strict=True)
    print("Loaded vision weights:", args.vision_checkpoint)
    print("Loaded GT-pose weights:", args.pose_checkpoint)
    del pose_saved

    n_params = sum(p.numel()for p in model.parameters())

    print(f"Model parameters: {n_params / 1e6:.2f} M")

    # Keep a reference to the normal, uncompiled model.
    # Use this for direct attribute access and checkpoint saving.
    raw_model = model

    # Keep one optimizer across phases: pose momentum survives the transition.
    vision_parameters = (list(model.backbone.parameters())
                         + list(model.vision_head.parameters())
                         + list(model.corr_decoder.parameters()))
    optimizer = torch.optim.AdamW(
        [{"params": vision_parameters, "lr": args.lr, "name": "vision"},
         {"params": list(model.pose_head.parameters()), "lr": args.adapt_lr, "name": "pose"}],
        weight_decay=0.05,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    metric_names = (
        "vision_occupancy", "vision_radius", "vision_depth", "vision_width",
    )
    history = {key: [] for key in (
        "epoch", "stage", "vision_lr", "pose_lr", "objective", "loss", "vision_loss",
        "correspondence_loss", "depth_structure_loss", "depth_delta_loss",
        "pose_loss", "translation_loss", "rotation_loss",
        "val_epoch", "val_stage", "val_objective", "val_total", "val_vision",
        "val_correspondence", "val_depth_structure", "val_depth_delta",
        "val_pose", "val_translation", "val_rotation",
    )}
    weights = dict(vision_occupancy=args.lambda_occ, vision_radius=args.lambda_radius,
                   vision_depth=1.0, vision_width=args.lambda_width)
    for component in metric_names:
        for prefix in ("", "val_"):
            history[prefix + component] = []
            history[prefix + component + "_weighted"] = []
    best_val_loss = float("inf")  # Joint stage only.
    best_adapt_pose = float("inf")
    previous_stage = None
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        pose_only = epoch <= args.adapt_epochs
        stage = "pose_adapt" if pose_only else "joint"
        set_training_stage(model, optimizer, pose_only, args)
        if stage != previous_stage:
            print(f"\nStage: {stage}; vision lr={optimizer.param_groups[0]['lr']:g}; "
                  f"pose lr={optimizer.param_groups[1]['lr']:g}", flush=True)
            previous_stage = stage
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, args.occ_thresh,
            args.lambda_occ, args.lambda_radius, args.lambda_corr,
            args.lambda_depth_delta, args.lambda_pose, args.lambda_width,
            scaler, pose_only=pose_only,
        )
        history["epoch"].append(epoch)
        history["stage"].append(stage)
        history["vision_lr"].append(optimizer.param_groups[0]["lr"])
        history["pose_lr"].append(optimizer.param_groups[1]["lr"])
        train_mapping = {
            "objective": "objective", "loss": "total", "vision_loss": "vision",
            "correspondence_loss": "correspondence", "depth_structure_loss": "depth_structure",
            "depth_delta_loss": "depth_delta", "pose_loss": "pose",
            "translation_loss": "translation", "rotation_loss": "rotation",
        }
        for key, metric in train_mapping.items():
            history[key].append(train_metrics[metric])
        for component, weight in weights.items():
            history[component].append(train_metrics[component])
            history[component + "_weighted"].append(weight * train_metrics[component])
        log = (
            f"Epoch {epoch} [{stage}]: obj={train_metrics['objective']:.4f} | "
            f"tot={train_metrics['total']:.4f} | vis={train_metrics['vision']:.4f} | "
            f"occ={train_metrics['vision_occupancy']:.4f} | rad={train_metrics['vision_radius']:.4f} | "
            f"depth={train_metrics['vision_depth']:.4f} | width={train_metrics['vision_width']:.4f} | "
            f"corr={train_metrics['correspondence']:.4f} | struct={train_metrics['depth_structure']:.4f} | "
            f"delta={train_metrics['depth_delta']:.4f} | pose={train_metrics['pose']:.4f}"
        )
        improved = improved_adapt = False
        should_validate = (epoch == 1 or epoch % args.val_interval == 0
                           or epoch in (args.adapt_epochs, args.adapt_epochs + 1, args.epochs))
        if should_validate:
            val_metrics = validate(
                model, val_loader, device, args.occ_thresh, args.lambda_occ,
                args.lambda_radius, args.lambda_corr, args.lambda_depth_delta,
                args.lambda_pose, args.lambda_width, pose_only=pose_only,
            )
            history["val_epoch"].append(epoch)
            history["val_stage"].append(stage)
            for key, metric in {
                "val_objective": "objective", "val_total": "total", "val_vision": "vision",
                "val_correspondence": "correspondence", "val_depth_structure": "depth_structure",
                "val_depth_delta": "depth_delta", "val_pose": "pose",
                "val_translation": "translation", "val_rotation": "rotation",
            }.items():
                history[key].append(val_metrics[metric])
            for component, weight in weights.items():
                history["val_" + component].append(val_metrics[component])
                history["val_" + component + "_weighted"].append(weight * val_metrics[component])
            log += (f" | val_tot={val_metrics['total']:.4f} | val_vis={val_metrics['vision']:.4f}"
                    f" | val_occ={val_metrics['vision_occupancy']:.4f} | val_rad={val_metrics['vision_radius']:.4f}"
                    f" | val_depth={val_metrics['vision_depth']:.4f} | val_width={val_metrics['vision_width']:.4f}"
                    f" | val_pose={val_metrics['pose']:.4f}")
            if pose_only:
                improved_adapt = val_metrics["pose"] < best_adapt_pose
                best_adapt_pose = min(best_adapt_pose, val_metrics["pose"])
            else:
                improved = val_metrics["total"] < best_val_loss
                best_val_loss = min(best_val_loss, val_metrics["total"])
            if improved or improved_adapt:
                log += " | BEST ADAPT" if pose_only else " | BEST JOINT"
        state = {
            "epoch": epoch, "training_stage": stage,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "scaler_state_dict": scaler.state_dict(),
            "args": vars(args), "history": history,
            "best_val_loss": best_val_loss, "best_adapt_pose": best_adapt_pose,
        }
        torch.save(state, os.path.join(args.output_dir, "checkpoints", "latest.pt"))
        if improved:
            torch.save(state, os.path.join(args.output_dir, "checkpoints", "best.pt"))
        if improved_adapt:
            torch.save(state, os.path.join(args.output_dir, "checkpoints", "best_adapt.pt"))
        with open(os.path.join(args.output_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        print(log + f" | time={time.perf_counter() - epoch_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
