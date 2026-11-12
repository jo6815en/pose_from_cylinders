"""Vision pretraining in two stages, using the existing joint model.

--stage occupancy: train backbone/shared cylinder features + occupancy head
from scratch, BCE only. Radius/depth/width heads, corr decoder and pose are frozen.
--stage geometry --init-checkpoint PATH: load vision weights from occupancy
pretraining, use a NEW optimizer, and train the full original vision objective:
occupancy + radius + depth + width + correspondence + depth structure/delta.
Occupancy remains supervised; all vision parameters can adapt. Pose stays frozen.
--stage full (default): original train_vision.py behavior, trained from scratch.
No model architecture or loss weight changes are made by selecting a stage.
"""

import argparse
import json
import os
import random
import numpy as np
import time

import torch
import torch.nn.functional as F
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
    lambda_depth,
    lambda_corr,
    lambda_depth_struct,
    lambda_depth_delta,
    lambda_width,
    scaler,
    occupancy_only=False,
):
    model.train()

    totals = {
        "total": 0.0,
        "vision": 0.0,
        "vision_occupancy": 0.0,
        "vision_radius": 0.0,
        "vision_depth": 0.0,
        "vision_width": 0.0,
        "correspondence": 0.0,
        "depth_structure": 0.0,
        "depth_delta": 0.0,
    }

    for batch in loader:
        (
            img_a,
            vision_a,
            img_b,
            vision_b,
            _pose_ab,
        ) = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            outputs = model(
                img_a, img_b, return_corr=not occupancy_only, compute_pose=False,
            )
            pred_vision_a, pred_vision_b, _ = outputs[:3]
            corr_a, corr_b = outputs[3:] if not occupancy_only else (None, None)

            if occupancy_only:
                with torch.autocast(device_type=device.type, enabled=False):
                    occ_a = F.binary_cross_entropy(
                        pred_vision_a[..., 0].float().clamp(1e-6, 1.0 - 1e-6), vision_a[..., 0].float())
                    occ_b = F.binary_cross_entropy(
                        pred_vision_b[..., 0].float().clamp(1e-6, 1.0 - 1e-6), vision_b[..., 0].float())
               
                vision_loss_total = lambda_occ * 0.5 * (occ_a + occ_b)
                rad_a = rad_b = dep_a = dep_b = width_a = width_b = vision_loss_total.new_zeros(())
                corr_loss = depth_structure_loss = depth_delta_loss = vision_loss_total.new_zeros(())
            
            else:
                vis_a, occ_a, rad_a, dep_a, width_a = vision_loss(pred_vision_a, vision_a, occ_thresh=occ_thresh, lambda_occ=lambda_occ,
                    lambda_radius=lambda_radius, lambda_depth=lambda_depth, lambda_width=lambda_width)

                vis_b, occ_b, rad_b, dep_b, width_b = vision_loss(pred_vision_b, vision_b, occ_thresh=occ_thresh, lambda_occ=lambda_occ,
                    lambda_radius=lambda_radius, lambda_depth=lambda_depth, lambda_width=lambda_width)

                vision_loss_total = 0.5 * ( vis_a + vis_b)

                corr_loss = patch_correspondence_loss(corr_a, vision_a, corr_b, vision_b)

                depth_structure_loss = 0.5 * (relative_depth_structure_loss(pred_vision_a, vision_a,  occ_thresh)
                    + relative_depth_structure_loss(pred_vision_b, vision_b, occ_thresh))

                depth_delta_loss = matched_depth_delta_loss(pred_vision_a, vision_a, pred_vision_b, vision_b, occ_thresh)


            loss = (
                vision_loss_total
                + lambda_corr * corr_loss
                + lambda_depth_struct * depth_structure_loss
                + lambda_depth_delta * depth_delta_loss
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        totals["total"] += loss.item()
        totals["vision"] += vision_loss_total.item()
        totals["vision_occupancy"] += (0.5 * (occ_a + occ_b)).item()
        totals["vision_radius"] += (0.5 * (rad_a + rad_b)).item()
        totals["vision_depth"] += (0.5 * (dep_a + dep_b)).item()
        totals["vision_width"] += (0.5 * (width_a + width_b)).item()
        totals["correspondence"] += corr_loss.item()
        totals["depth_structure"] += depth_structure_loss.item()
        totals["depth_delta"] += depth_delta_loss.item()

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
    lambda_depth,
    lambda_corr,
    lambda_depth_struct,
    lambda_depth_delta,
    lambda_width,
    occupancy_only=False,
):
    model.eval()

    totals = {
        "total": 0.0,
        "vision": 0.0,
        "vision_occupancy": 0.0,
        "vision_radius": 0.0,
        "vision_depth": 0.0,
        "vision_width": 0.0,
        "correspondence": 0.0,
        "depth_structure": 0.0,
        "depth_delta": 0.0,
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
            _pose_ab,
        ) = move_batch_to_device(batch, device)

        outputs = model(
            img_a, img_b, return_corr=not occupancy_only, compute_pose=False,
        )
        pred_vision_a, pred_vision_b, _ = outputs[:3]
        corr_a, corr_b = outputs[3:] if not occupancy_only else (None, None)

        if occupancy_only:
            with torch.autocast(device_type=device.type, enabled=False):
                occ_a = F.binary_cross_entropy(
                    pred_vision_a[..., 0].float().clamp(1e-6, 1.0 - 1e-6), vision_a[..., 0].float())
                occ_b = F.binary_cross_entropy(
                    pred_vision_b[..., 0].float().clamp(1e-6, 1.0 - 1e-6), vision_b[..., 0].float())
            vision_loss_total = lambda_occ * 0.5 * (occ_a + occ_b)
            rad_a = rad_b = dep_a = dep_b = width_a = width_b = vision_loss_total.new_zeros(())
            corr_loss = depth_structure_loss = depth_delta_loss = vision_loss_total.new_zeros(())
        else:
            vis_a, occ_a, rad_a, dep_a, width_a = vision_loss(pred_vision_a, vision_a, occ_thresh=occ_thresh, lambda_occ=lambda_occ,
                lambda_radius=lambda_radius, lambda_depth=lambda_depth, lambda_width=lambda_width)
            vis_b, occ_b, rad_b, dep_b, width_b = vision_loss(pred_vision_b, vision_b, occ_thresh=occ_thresh, lambda_occ=lambda_occ,
                lambda_radius=lambda_radius, lambda_depth=lambda_depth, lambda_width=lambda_width)
            vision_loss_total = 0.5 * (vis_a + vis_b)

            corr_loss = patch_correspondence_loss(corr_a, vision_a, corr_b, vision_b)

            depth_structure_loss = 0.5 * (
                relative_depth_structure_loss(pred_vision_a, vision_a, occ_thresh)
                + relative_depth_structure_loss(pred_vision_b, vision_b, occ_thresh)
            )

            depth_delta_loss = matched_depth_delta_loss(pred_vision_a, vision_a, pred_vision_b, vision_b, occ_thresh)

        total_loss = (
            vision_loss_total
            + lambda_corr * corr_loss
            + lambda_depth_struct * depth_structure_loss
            + lambda_depth_delta * depth_delta_loss
        )

        if not occupancy_only:
            radius_a, position_a = relative_cylinder_errors(
                pred_vision_a, vision_a, occ_thresh
            )
            radius_b, position_b = relative_cylinder_errors(
                pred_vision_b, vision_b, occ_thresh
            )

        totals["total"] += total_loss.item()
        totals["vision"] += vision_loss_total.item()
        totals["vision_occupancy"] += (0.5 * (occ_a + occ_b)).item()
        totals["vision_radius"] += (0.5 * (rad_a + rad_b)).item()
        totals["vision_depth"] += (0.5 * (dep_a + dep_b)).item()
        totals["vision_width"] += (0.5 * (width_a + width_b)).item()
        totals["correspondence"] += corr_loss.item()
        totals["depth_structure"] += depth_structure_loss.item()
        totals["depth_delta"] += depth_delta_loss.item()
        if not occupancy_only:
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        default="/nobackup/proj/disk/midlevel_representations/personal/johanna/data",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
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
    parser.add_argument("--lambda-depth", type=float, default=1.0)
    parser.add_argument("--lambda-width", type=float, default=0.1)
    parser.add_argument("--lambda-corr", type=float, default=1.0)
    parser.add_argument("--lambda-depth-delta", type=float, default=0.35)
    parser.add_argument("--lambda-depth-struct", type=float, default=0.5)


    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--stage", choices=("full", "occupancy", "geometry"), default="full",
                        help="full: original vision training; occupancy: BCE only; geometry: continue from occupancy with the full vision objective")
    parser.add_argument("--init-checkpoint", default=None,
                        help="Required for --stage geometry: occupancy-pretrained checkpoint")

    args = parser.parse_args()
    args.occupancy_only = args.stage == "occupancy"
    if args.stage == "geometry" and not args.init_checkpoint:
        parser.error("--stage geometry requires --init-checkpoint")
    if args.init_checkpoint and args.stage != "geometry":
        parser.error("--init-checkpoint is only used with --stage geometry")
    args.output_dir = args.output_dir or {
        "full": "runs/vision_only", "occupancy": "runs/vision_occupancy",
        "geometry": "runs/vision_geometry",
    }[args.stage]
    if min(args.epochs, args.batch_size, args.val_interval, args.lr, args.lambda_occ) <= 0:
        parser.error("Epochs, batch size, validation interval, learning rate and occupancy weight must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")

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
    ).to(device)

    if args.init_checkpoint:
        saved = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        saved_cfg = saved.get("args", {})
        for key in ("img_size", "patch_size", "embed_dim", "depth", "num_heads", "num_bins", "dropout"):
            if key in saved_cfg:
                old, new = saved_cfg[key], getattr(args, key)
                if key == "patch_size":
                    old, new = tuple(old), tuple(new)
                if old != new:
                    parser.error(f"Checkpoint {key}={old} differs from {new}; set --{key.replace('_', '-')} to match")
        model.load_vision_checkpoint(args.init_checkpoint)
        print("Loaded occupancy weights:", args.init_checkpoint, "epoch", saved.get("epoch"))
        del saved

    # Always train shared image features; never train the pose head here.
    model.backbone.requires_grad_(True)
    model.vision_head.requires_grad_(True)
    model.pose_head.requires_grad_(False)
    model.corr_decoder.requires_grad_(not args.occupancy_only)
    if args.occupancy_only:
        for name in ("radius_head", "depth_head", "width_head"):
            getattr(model.vision_head, name).requires_grad_(False)
    print("Vision training stage:", args.stage)

    n_params = sum(p.numel()for p in model.parameters())

    print(f"Model parameters: {n_params / 1e6:.2f} M")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Keep a reference to the normal, uncompiled model.
    # Use this for direct attribute access and checkpoint saving.
    raw_model = model

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=0.05,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Epoch-aligned history dict. This format is directly plottable and is
    # stored unchanged in history.json, latest.pt, and best.pt.
    history = {
        "epoch": [],
        "loss": [],
        "vision_loss": [],
        "correspondence_loss": [],
        "depth_structure_loss": [],
        "depth_delta_loss": [],
        "vision_occupancy": [], "vision_radius": [], "vision_depth": [], "vision_width": [],
        "vision_occupancy_weighted": [], "vision_radius_weighted": [],
        "vision_depth_weighted": [], "vision_width_weighted": [],
        "val_vision_occupancy": [], "val_vision_radius": [], "val_vision_depth": [], "val_vision_width": [],
        "val_vision_occupancy_weighted": [], "val_vision_radius_weighted": [],
        "val_vision_depth_weighted": [], "val_vision_width_weighted": [],
        "val_epoch": [],
        "val_total": [],
        "val_vision": [],
        "val_correspondence": [],
        "val_depth_structure": [],
        "val_depth_delta": [],
    }
    if args.occupancy_only:
        keep = {"epoch", "loss", "vision_loss", "vision_occupancy", "vision_occupancy_weighted",
                "val_epoch", "val_total", "val_vision", "val_vision_occupancy", "val_vision_occupancy_weighted"}
        history = {key: values for key, values in history.items() if key in keep}
    best_val_loss = float("inf")
    best_val_vision = float("inf")
    vision_weights = dict(vision_occupancy=args.lambda_occ,
                          vision_radius=args.lambda_radius,
                          vision_depth=1.0, vision_width=args.lambda_width)
    if args.occupancy_only:
        vision_weights = {key: (weight if key == "vision_occupancy" else 0.0)
                          for key, weight in vision_weights.items()}

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.occ_thresh,
            args.lambda_occ,
            args.lambda_radius,
            args.lambda_depth,
            args.lambda_corr,
            args.lambda_depth_struct,
            args.lambda_depth_delta,
            args.lambda_width,
            scaler,
            occupancy_only=args.occupancy_only,
        )

        history["epoch"].append(epoch)
        history["loss"].append(train_metrics["total"])
        history["vision_loss"].append(train_metrics["vision"])
        for component, weight in vision_weights.items():
            if component in history:
                history[component].append(train_metrics[component])
                history[component + "_weighted"].append(weight * train_metrics[component])
        if "correspondence_loss" in history:
            history["correspondence_loss"].append(train_metrics["correspondence"])
        if "depth_structure_loss" in history:
            history["depth_structure_loss"].append(train_metrics["depth_structure"])
        if "depth_delta_loss" in history:
            history["depth_delta_loss"].append(train_metrics["depth_delta"])

        log = (
            f"Epoch {epoch}: "
            f"tot={train_metrics['total']:.4f} | "
            f"vis={train_metrics['vision']:.4f} | "
            f"occ={train_metrics['vision_occupancy']:.4f} | "
            f"rad={train_metrics['vision_radius']:.4f} | "
            f"depth={train_metrics['vision_depth']:.4f} | "
            f"width={train_metrics['vision_width']:.4f} | "
            f"corr={train_metrics['correspondence']:.4f} | "
            f"struct={train_metrics['depth_structure']:.4f} | "
            f"delta={train_metrics['depth_delta']:.4f} | "
        )

        if args.occupancy_only:
            log = (f"Epoch {epoch} [occupancy]: tot={train_metrics['total']:.4f} | "
                   f"occ={train_metrics['vision_occupancy']:.4f}")

        if epoch == 1 or epoch % args.val_interval == 0:
            val_metrics = validate(
                model,
                val_loader,
                device,
                args.occ_thresh,
                args.lambda_occ,
                args.lambda_radius,
                args.lambda_depth,
                args.lambda_corr,
                args.lambda_depth_struct,
                args.lambda_depth_delta,
                args.lambda_width,
                occupancy_only=args.occupancy_only,
            )

            history["val_epoch"].append(epoch)
            history["val_total"].append(val_metrics["total"])
            history["val_vision"].append(val_metrics["vision"])
            for component, weight in vision_weights.items():
                if "val_" + component in history:
                    history["val_" + component].append(val_metrics[component])
                if "val_" + component + "_weighted" in history:
                    history["val_" + component + "_weighted"].append(weight * val_metrics[component])
            if "val_correspondence" in history:
                history["val_correspondence"].append(val_metrics["correspondence"])
            if "val_depth_structure" in history:
                history["val_depth_structure"].append(val_metrics["depth_structure"])
            if "val_depth_delta" in history:
                history["val_depth_delta"].append(val_metrics["depth_delta"])

            log += (
                f" | val_tot={val_metrics['total']:.4f}"
                f" | val_vis={val_metrics['vision']:.4f}"
                f" | val_occ={val_metrics['vision_occupancy']:.4f}"
                f" | val_rad={val_metrics['vision_radius']:.4f}"
                f" | val_depth={val_metrics['vision_depth']:.4f}"
                f" | val_width={val_metrics['vision_width']:.4f}"
                f" | val_corr={val_metrics['correspondence']:.4f}"
                f" | val_struct={val_metrics['depth_structure']:.4f}"
                f" | val_delta={val_metrics['depth_delta']:.4f}"
            )
            if args.occupancy_only:
                log = (f"Epoch {epoch} [occupancy]: tot={train_metrics['total']:.4f} | "
                       f"occ={train_metrics['vision_occupancy']:.4f} | "
                       f"val_tot={val_metrics['total']:.4f} | val_occ={val_metrics['vision_occupancy']:.4f}")
            if val_metrics["total"] < best_val_loss:
                best_val_loss = val_metrics["total"]

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": raw_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "args": vars(args),
                        "training_stage": "vision_" + args.stage,
                        "history": history,
                        "best_val_loss": best_val_loss,
                    },
                    os.path.join(
                        args.output_dir,
                        "checkpoints",
                        "best.pt",
                    ),
                )

                log += f" | BEST (val_tot={best_val_loss:.4f})"

            if val_metrics["vision"] < best_val_vision:
                best_val_vision = val_metrics["vision"]
                torch.save(
                    {"epoch": epoch, "model_state_dict": raw_model.state_dict(),
                     "optimizer_state_dict": optimizer.state_dict(), "args": vars(args),
                        "training_stage": "vision_" + args.stage,
                     "history": history, "best_val_vision": best_val_vision,
                     "best_val_loss": best_val_loss},
                    os.path.join(args.output_dir, "checkpoints", "best_vision.pt"),
                )
                log += " | BEST VISION"

        epoch_time = time.perf_counter() - epoch_start
        log += f" | time={epoch_time:.1f}s"
        print(log, flush=True)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                        "training_stage": "vision_" + args.stage,
                "history": history,
            },
            os.path.join(
                args.output_dir,
                "checkpoints",
                "latest.pt",
            ),
        )

        with open(
            os.path.join(args.output_dir, "history.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
