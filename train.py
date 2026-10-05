import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from dataset import SceneTwoPairsDataset
from augmentations import build_train_transform, build_eval_transform
from model import PairImageCylinderModel
from losses import (
    vision_loss,
    patch_correspondence_loss,
    relative_depth_structure_loss,
)
from metrics import relative_cylinder_errors
from utils import save_checkpoint


def move_batch_to_device(batch, device):
    return tuple(x.to(device, non_blocking=True) for x in batch)


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    scaler,
    occ_thresh,
    lambda_occ,
    lambda_radius,
    lambda_corr,
    lambda_depth_structure,
):
    model.train()

    totals = {
        "total": 0.0,
        "supervised": 0.0,
        "vision": 0.0,
        "correspondence": 0.0,
        # Kept for compatibility with the notebook history.
        "pose": 0.0,
        "translation_error": 0.0,
        "translation_magnitude_error": 0.0,
        "translation_direction_error": 0.0,
        "yaw_error": 0.0,
        "radius_consistency": 0.0,
        "reprojection": 0.0,
    }

    if len(loader) == 0:
        raise RuntimeError("Training loader is empty.")

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)

        (
            img_a1,
            vision_a1,
            img_b1,
            vision_b1,
            _pose_ab1,
            img_a2,
            vision_a2,
            img_b2,
            vision_b2,
            _pose_ab2,
        ) = move_batch_to_device(batch, device)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            # Exactly as train_v2.ipynb: PoseHead is not computed/trained.
            pred_vision_a1, pred_vision_b1, _, corr_a1, corr_b1 = model(
                img_a1,
                img_b1,
                compute_pose=False,
                return_corr=True,
            )
            pred_vision_a2, pred_vision_b2, _, corr_a2, corr_b2 = model(
                img_a2,
                img_b2,
                compute_pose=False,
                return_corr=True,
            )

            corr1 = patch_correspondence_loss(
                corr_a1, vision_a1, corr_b1, vision_b1
            )
            corr2 = patch_correspondence_loss(
                corr_a2, vision_a2, corr_b2, vision_b2
            )
            corr_loss = 0.5 * (corr1 + corr2)

            vis_a1, *_ = vision_loss(
                pred_vision_a1,
                vision_a1,
                occ_thresh=occ_thresh,
                lambda_occ=lambda_occ,
                lambda_radius=lambda_radius,
                lambda_depth=1.0,
            )
            vis_b1, *_ = vision_loss(
                pred_vision_b1,
                vision_b1,
                occ_thresh=occ_thresh,
                lambda_occ=lambda_occ,
                lambda_radius=lambda_radius,
                lambda_depth=1.0,
            )
            vis_a2, *_ = vision_loss(
                pred_vision_a2,
                vision_a2,
                occ_thresh=occ_thresh,
                lambda_occ=lambda_occ,
                lambda_radius=lambda_radius,
                lambda_depth=1.0,
            )
            vis_b2, *_ = vision_loss(
                pred_vision_b2,
                vision_b2,
                occ_thresh=occ_thresh,
                lambda_occ=lambda_occ,
                lambda_radius=lambda_radius,
                lambda_depth=1.0,
            )

            vision_loss_total = 0.25 * (
                vis_a1 + vis_b1 + vis_a2 + vis_b2
            )

            depth_structure1 = 0.5 * (
                relative_depth_structure_loss(
                    pred_vision_a1, vision_a1, occ_thresh
                )
                + relative_depth_structure_loss(
                    pred_vision_b1, vision_b1, occ_thresh
                )
            )
            depth_structure2 = 0.5 * (
                relative_depth_structure_loss(
                    pred_vision_a2, vision_a2, occ_thresh
                )
                + relative_depth_structure_loss(
                    pred_vision_b2, vision_b2, occ_thresh
                )
            )
            depth_structure_loss = 0.5 * (
                depth_structure1 + depth_structure2
            )

            loss = (
                vision_loss_total
                + lambda_corr * corr_loss
                + lambda_depth_structure * depth_structure_loss
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        totals["total"] += loss.item()
        totals["supervised"] += vision_loss_total.item()
        totals["vision"] += vision_loss_total.item()
        totals["correspondence"] += corr_loss.item()

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
    lambda_depth_structure,
):
    """Validation uses the same objective as training:
    vision + lambda_corr * correspondence
    + lambda_depth_structure * relative depth structure.
    """
    model.eval()

    totals = {
        "total": 0.0,
        "supervised": 0.0,
        "vision": 0.0,
        "correspondence": 0.0,
        "pose": 0.0,
        "radius_consistency": 0.0,
        "reprojection": 0.0,
        "translation_error": 0.0,
        "translation_magnitude_error": 0.0,
        "translation_direction_error": 0.0,
        "yaw_error": 0.0,
        "cylinder_radius_rel_l2": 0.0,
        "cylinder_position_rel_l2": 0.0,
        "camera_translation_rel_l2": 0.0,
        "camera_rotation_rel_l2": 0.0,
    }

    num_batches = 0

    for batch in loader:
        img_a, vision_a, img_b, vision_b, _pose_ab = move_batch_to_device(
            batch, device
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            pred_vision_a, pred_vision_b, _, corr_a, corr_b = model(
                img_a,
                img_b,
                compute_pose=False,
                return_corr=True,
            )

            vis_a, *_ = vision_loss(
                pred_vision_a,
                vision_a,
                occ_thresh=occ_thresh,
                lambda_occ=lambda_occ,
                lambda_radius=lambda_radius,
                lambda_depth=1.0,
            )
            vis_b, *_ = vision_loss(
                pred_vision_b,
                vision_b,
                occ_thresh=occ_thresh,
                lambda_occ=lambda_occ,
                lambda_radius=lambda_radius,
                lambda_depth=1.0,
            )
            vision_loss_total = 0.5 * (vis_a + vis_b)

            corr_loss = patch_correspondence_loss(
                corr_a, vision_a, corr_b, vision_b
            )

            depth_structure_loss = 0.5 * (
                relative_depth_structure_loss(
                    pred_vision_a, vision_a, occ_thresh
                )
                + relative_depth_structure_loss(
                    pred_vision_b, vision_b, occ_thresh
                )
            )

            # Same objective and weights as training.
            total_loss = (
                vision_loss_total
                + lambda_corr * corr_loss
                + lambda_depth_structure * depth_structure_loss
            )

        radius_a, position_a = relative_cylinder_errors(
            pred_vision_a, vision_a, occ_thresh
        )
        radius_b, position_b = relative_cylinder_errors(
            pred_vision_b, vision_b, occ_thresh
        )

        totals["cylinder_radius_rel_l2"] += (
            0.5 * (radius_a + radius_b)
        ).item()
        totals["cylinder_position_rel_l2"] += (
            0.5 * (position_a + position_b)
        ).item()

        totals["total"] += total_loss.item()
        totals["supervised"] += vision_loss_total.item()
        totals["vision"] += vision_loss_total.item()
        totals["correspondence"] += corr_loss.item()
        num_batches += 1

    if num_batches == 0:
        raise RuntimeError(
            "Validation loader is empty; cannot select a best model."
        )

    return {key: value / num_batches for key, value in totals.items()}


def make_history():
    return {
        "epoch": [],
        "total": [],
        "supervised": [],
        "vision": [],
        "pose": [],
        "translation_error": [],
        "correspondence": [],
        "translation_magnitude_error": [],
        "translation_direction_error": [],
        "yaw_error_deg": [],
        "radius_consistency": [],
        "reprojection": [],
        "forest_epoch": [],
        "forest_total": [],
        "forest_radius_consistency": [],
        "forest_reprojection": [],
        "forest_sparsity": [],
        "val_epoch": [],
        "val_total": [],
        "val_supervised": [],
        "val_vision": [],
        "val_pose": [],
        "val_translation_error": [],
        "val_translation_magnitude_error": [],
        "val_translation_direction_error": [],
        "val_yaw_error_deg": [],
        "val_radius_consistency": [],
        "val_reprojection": [],
        "val_cylinder_radius_rel_l2": [],
        "val_cylinder_position_rel_l2": [],
        "val_camera_translation_rel_l2": [],
        "val_camera_rotation_rel_l2": [],
        "val_correspondence": [],
    }


def append_train_history(history, epoch, metrics):
    history["epoch"].append(epoch)
    history["total"].append(metrics["total"])
    history["supervised"].append(metrics["supervised"])
    history["vision"].append(metrics["vision"])
    history["pose"].append(metrics["pose"])
    history["translation_error"].append(metrics["translation_error"])
    history["correspondence"].append(metrics["correspondence"])
    history["translation_magnitude_error"].append(
        metrics["translation_magnitude_error"]
    )
    history["translation_direction_error"].append(
        metrics["translation_direction_error"]
    )
    history["yaw_error_deg"].append(metrics["yaw_error"])
    history["radius_consistency"].append(metrics["radius_consistency"])
    history["reprojection"].append(metrics["reprojection"])


def append_val_history(history, epoch, metrics):
    history["val_epoch"].append(epoch)
    history["val_total"].append(metrics["total"])
    history["val_supervised"].append(metrics["supervised"])
    history["val_vision"].append(metrics["vision"])
    history["val_correspondence"].append(metrics["correspondence"])
    history["val_pose"].append(metrics["pose"])
    history["val_translation_error"].append(metrics["translation_error"])
    history["val_translation_magnitude_error"].append(
        metrics["translation_magnitude_error"]
    )
    history["val_translation_direction_error"].append(
        metrics["translation_direction_error"]
    )
    history["val_yaw_error_deg"].append(metrics["yaw_error"])
    history["val_radius_consistency"].append(metrics["radius_consistency"])
    history["val_reprojection"].append(metrics["reprojection"])
    history["val_cylinder_radius_rel_l2"].append(
        metrics["cylinder_radius_rel_l2"]
    )
    history["val_cylinder_position_rel_l2"].append(
        metrics["cylinder_position_rel_l2"]
    )
    history["val_camera_translation_rel_l2"].append(
        metrics["camera_translation_rel_l2"]
    )
    history["val_camera_rotation_rel_l2"].append(
        metrics["camera_rotation_rel_l2"]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Cluster training matching train_v2.ipynb."
    )
    parser.add_argument(
        "--data-dir",
        default="/nobackup/proj/disk/midlevel_representations/personal/johanna/data",
        help="Directory containing dataset/ and valdataset/.",
    )
    parser.add_argument("--output-dir", default="runs/train_v2")

    # Defaults match train_v2.ipynb.
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--val-interval", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(16, 8),
        metavar=("H", "W"),
    )
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-bins", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--lambda-occ", type=float, default=10.0)
    parser.add_argument("--lambda-radius", type=float, default=10.0)
    parser.add_argument("--occ-thresh", type=float, default=0.5)

    # Effective training value is 1.0.
    parser.add_argument("--lambda-corr", type=float, default=1.0)
    parser.add_argument("--lambda-depth-structure", type=float, default=0.5)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda")
    )

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_model_path = os.path.join(checkpoint_dir, "best_total.pt")
    latest_model_path = os.path.join(checkpoint_dir, "latest_model.pt")
    history_path = os.path.join(args.output_dir, "history.json")

    print("CUDA available:", torch.cuda.is_available(), flush=True)
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0), flush=True)

    model = PairImageCylinderModel(
        img_size=args.img_size,
        patch_size=tuple(args.patch_size),
        in_chans=3,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        num_bins=args.num_bins,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.05,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"Model parameters: {n_params:,} ({n_params / 1e6:.2f}M)",
        flush=True,
    )

    train_dataset = SceneTwoPairsDataset(
        root_dir=os.path.join(args.data_dir, "dataset"),
        image_size=args.img_size,
        debug=False,
        return_two_pairs=True,
        transform=build_train_transform(args.img_size),
    )
    val_dataset = SceneTwoPairsDataset(
        root_dir=os.path.join(args.data_dir, "valdataset"),
        image_size=args.img_size,
        debug=False,
        return_two_pairs=False,
        transform=build_eval_transform(args.img_size),
    )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": (device.type == "cuda"),
        "persistent_workers": (args.num_workers > 0),
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    print("Train batches:", len(train_loader), flush=True)
    print("Val batches:", len(val_loader), flush=True)

    history = make_history()
    best_val_loss = float("inf")
    best_epoch = None

    for epoch in range(args.epochs):
        epoch_num = epoch + 1

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            occ_thresh=args.occ_thresh,
            lambda_occ=args.lambda_occ,
            lambda_radius=args.lambda_radius,
            lambda_corr=args.lambda_corr,
            lambda_depth_structure=args.lambda_depth_structure,
        )
        append_train_history(history, epoch_num, train_metrics)

        log = (
            f"Epoch {epoch_num}: "
            f"tot={train_metrics['total']:.4f} | "
            f"vis={train_metrics['vision']:.4f} | "
            f"corr={train_metrics['correspondence']:.4f}"
        )

        if epoch_num == 1 or epoch_num % args.val_interval == 0:
            val_metrics = validate(
                model=model,
                loader=val_loader,
                device=device,
                occ_thresh=args.occ_thresh,
                lambda_occ=args.lambda_occ,
                lambda_radius=args.lambda_radius,
                lambda_corr=args.lambda_corr,
                lambda_depth_structure=args.lambda_depth_structure,
            )
            append_val_history(history, epoch_num, val_metrics)

            is_best_total = val_metrics["total"] < best_val_loss
            if is_best_total:
                best_val_loss = val_metrics["total"]
                best_epoch = epoch_num
                save_checkpoint(
                    best_model_path,
                    model,
                    optimizer,
                    history,
                    epoch_num,
                    val_metrics,
                )

            log += (
                f" | val_tot={val_metrics['total']:.4f}"
                f" | val_vis={val_metrics['vision']:.4f}"
                f" | val_corr={val_metrics['correspondence']:.4f}"
            )
            if is_best_total:
                log += " | BEST TOTAL"

        # Same state as the notebook latest checkpoint, with args added for
        # cluster reproducibility.
        torch.save(
            {
                "epoch": epoch_num,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "history": history,
                "args": vars(args),
            },
            latest_model_path,
        )

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        print(log, flush=True)

    print(
        f"Best total: {best_val_loss:.6f} at epoch {best_epoch}",
        flush=True,
    )
    print(
        f"Latest model and history saved to: {latest_model_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
