"""GT-bin input ablation for the existing model_pose architecture.

Inputs: occupancy, log1p(radius), log1p(depth), and a GT correspondence map.
The correspondence map is derived from matching cylinder IDs across views.
In each input its columns follow the current view's bins and its rows follow
bins in the other view. No raw IDs or pose targets are passed as features.
The existing model_pose uses in_chans=4; all other architecture settings remain.
Inference must use this file's predict_pose_from_gt_bins with a 4-channel model.
Training: existing backbone + pose_head, pose loss only. Other heads frozen.
The dataset still loads images, but images are never passed to the model.
Legacy loss-weight CLI options remain accepted but are unused in this ablation.
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
from losses import relative_pose_loss_2d
from augmentations import build_train_transform, build_eval_transform
from model_pose import PairImageCylinderModel


def move_batch_to_device(batch, device):
    return tuple(x.to(device, non_blocking=True) for x in batch)


def gt_bins_to_input(vision, image_size, correspondence):
    """GT geometry + correspondence [B, other_bins, current_bins] -> 4 channels."""
    occupied = (vision[..., 0] > 0.5).float()
    radius = vision[..., 1].float().clamp_min(0)
    depth = vision[..., 2].float().clamp_min(0)
    features = torch.stack([
        occupied,
        torch.log1p(radius) * occupied,
        torch.log1p(depth) * occupied,
    ], dim=1).unsqueeze(2)
    
    geometry = F.interpolate(
        features, size=(image_size, image_size), mode="nearest"
    )
    correspondence = F.interpolate(
        correspondence.float().unsqueeze(1),
        size=(image_size, image_size),
        mode="nearest",
    )
    return torch.cat([geometry, correspondence], dim=1)


def gt_correspondence_maps(vision_a, vision_b):
    """Return maps for A and B, using occupied GT bins with identical IDs.

    matches[b, ia, ib] is True exactly when A bin ia and B bin ib belong
    to the same cylinder. Occupancy excludes empty bins even if their ID is 0.
    This also supports a cylinder covering multiple bins.
    """
    occ_a = vision_a[..., 0] > 0.5
    occ_b = vision_b[..., 0] > 0.5
    ids_a = vision_a[..., 3].round().long()
    ids_b = vision_b[..., 3].round().long()
    matches = (
        (ids_a.unsqueeze(2) == ids_b.unsqueeze(1))
        & occ_a.unsqueeze(2)
        & occ_b.unsqueeze(1)
    )
    # Geometry varies along x. Keep x associated with the current view.
    map_a = matches.transpose(1, 2)  # y = B bins, x = A bins
    map_b = matches                  # y = A bins, x = B bins
    return map_a, map_b


def predict_pose_from_gt_bins(model, vision_a, vision_b):
    patch_embed = model.backbone.patch_embed
    image_size = patch_embed.grid_h * patch_embed.patch_h
    map_a, map_b = gt_correspondence_maps(vision_a, vision_b)
    if not getattr(model, "use_gt_correspondences", True):
        map_a = torch.zeros_like(map_a)
        map_b = torch.zeros_like(map_b)
    input_a = gt_bins_to_input(vision_a, image_size, map_a)
    input_b = gt_bins_to_input(vision_b, image_size, map_b)
    camera_a, camera_b, patches_a, patches_b = model.backbone(input_a, input_b)
    
    return model.pose_head(camera_a, camera_b, patches_a, patches_b)


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
):
    model.train()
    totals = {key: 0.0 for key in ("total", "pose", "translation", "rotation")}
    for batch in loader:
        img_a, vision_a, img_b, vision_b, pose_ab = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            pred_pose = predict_pose_from_gt_bins(model, vision_a, vision_b)
            
            pose_total, t_loss, r_loss = relative_pose_loss_2d(pred_pose, pose_ab, t_weight=1.0,r_weight=1.0)
            
            loss = pose_total
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        totals["total"] += loss.item()
        totals["pose"] += pose_total.item()
        totals["translation"] += t_loss.item()
        totals["rotation"] += r_loss.item()
    
    if len(loader) == 0:
        raise RuntimeError("Training loader is empty.")
    return {key: value / len(loader) for key, value in totals.items()}

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
):
    model.eval()
    totals = {key: 0.0 for key in ("total", "pose", "translation", "rotation")}
    num_batches = 0
    
    for batch in loader:
        img_a, vision_a, img_b, vision_b, pose_ab = move_batch_to_device(batch, device)
        
        pred_pose = predict_pose_from_gt_bins(model, vision_a, vision_b)

        pose_total, t_loss, r_loss = relative_pose_loss_2d(pred_pose, pose_ab, t_weight=1.0, r_weight=1.0)
        totals["total"] += pose_total.item()
        totals["pose"] += pose_total.item()
        totals["translation"] += t_loss.item()
        totals["rotation"] += r_loss.item()
        num_batches += 1
    
    if num_batches == 0:
        raise RuntimeError("Validation loader is empty.")
    
    return {key: value / num_batches for key, value in totals.items()}


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

    parser.add_argument("--lambda-occ", type=float, default=10.0)
    parser.add_argument("--lambda-radius", type=float, default=10.0)
    parser.add_argument("--occ-thresh", type=float, default=0.5)
    parser.add_argument("--lambda-corr", type=float, default=1.0)
    parser.add_argument("--lambda-depth-delta", type=float, default=0.35)
    parser.add_argument("--lambda-pose", type=float, default=2.0)
    parser.add_argument("--lambda-width", type=float, default=0.1)

    parser.add_argument(
        "--output-dir",
        default="runs/gt_bins_pose_correspondences",
    )

    parser.add_argument(
        "--no-gt-correspondences", dest="gt_correspondences",
        action="store_false", default=True,
        help="Control run: identical 4-channel architecture, but zero match maps.",
    )

    args = parser.parse_args()
    args.in_chans = 4
    args.input_representation = (
        "gt_bins_with_gt_correspondences" if args.gt_correspondences
        else "gt_bins_with_zero_correspondence_channel"
    )
    if args.num_bins != args.img_size:
        raise ValueError(
            "This correspondence-map ablation requires --num-bins == --img-size "
            "so resizing cannot drop a known bin correspondence."
        )

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
        in_chans=args.in_chans,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        num_bins=args.num_bins,
        dropout=args.dropout,
    ).to(device)

    model.use_gt_correspondences = args.gt_correspondences

    for parameter in model.vision_head.parameters():
        parameter.requires_grad_(False)
    for parameter in model.corr_decoder.parameters():
        parameter.requires_grad_(False)

    n_params = sum(p.numel()for p in model.parameters())

    print(f"Model parameters: {n_params / 1e6:.2f} M")

    # Keep a reference to the normal, uncompiled model.
    # Use this for direct attribute access and checkpoint saving.
    raw_model = model

    # Direct backbone/pose-head calls; compilation is disabled for this ablation.

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
        "pose": [],
        "translation": [],
        "rotation": [],
        "val_epoch": [],
        "val_pose": [],
        "val_translation": [],
        "val_rotation": [],
    }
    best_val_loss = float("inf")

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
            args.lambda_corr,
            args.lambda_depth_delta,
            args.lambda_pose,
            args.lambda_width,
            scaler,
        )


        history["epoch"].append(epoch)
        history["pose"].append(train_metrics["pose"])
        history["translation"].append(train_metrics["translation"])
        history["rotation"].append(train_metrics["rotation"])

        log = (
            f"Epoch {epoch}: "
            f"pose={train_metrics['pose']:.4f} | "
            f"translation={train_metrics['translation']:.4f} | "
            f"rotation={train_metrics['rotation']:.4f}"
        )

        if epoch == 1 or epoch % args.val_interval == 0:
            val_metrics = validate(
                model,
                val_loader,
                device,
                args.occ_thresh,
                args.lambda_occ,
                args.lambda_radius,
                args.lambda_corr,
                args.lambda_depth_delta,
                args.lambda_pose,
                args.lambda_width,
            )

            history["val_epoch"].append(epoch)
            history["val_pose"].append(val_metrics["pose"])
            history["val_translation"].append(val_metrics["translation"])
            history["val_rotation"].append(val_metrics["rotation"])

            log += (
                f" | val_tot={val_metrics['total']:.4f}"
                f" | val_pose={val_metrics['pose']:.4f}"
                f" | val_translation={val_metrics['translation']:.4f}"
                f" | val_rotation={val_metrics['rotation']:.4f}"
            )
            if val_metrics["total"] < best_val_loss:
                best_val_loss = val_metrics["total"]

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": raw_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "args": vars(args),
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

        epoch_time = time.perf_counter() - epoch_start
        log += f" | time={epoch_time:.1f}s"
        print(log, flush=True)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
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
