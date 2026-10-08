import argparse
import json
import os

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
    scaler,
):
    model.train()

    totals = {
        "total": 0.0,
        "supervised": 0.0,
        "vision": 0.0,
        "correspondence": 0.0,
        "depth_structure": 0.0,
        "depth_delta": 0.0,
        # Kept so the existing history/checkpoint structure remains compatible.
        "pose": 0.0,
        "translation_error": 0.0,
        "translation_magnitude_error": 0.0,
        "translation_direction_error": 0.0,
        "yaw_error_deg": 0.0,
        "radius_consistency": 0.0,
        "reprojection": 0.0,
    }

    for batch in loader:
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

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
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

            depth_delta1 = matched_depth_delta_loss(
                pred_vision_a1, vision_a1, pred_vision_b1, vision_b1, occ_thresh
            )
            depth_delta2 = matched_depth_delta_loss(
                pred_vision_a2, vision_a2, pred_vision_b2, vision_b2, occ_thresh
            )
            depth_delta_loss = 0.5 * (depth_delta1 + depth_delta2)

            loss = (
                vision_loss_total
                + lambda_corr * corr_loss
                + 0.5 * depth_structure_loss
                + lambda_depth_delta * depth_delta_loss
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        totals["total"] += loss.item()
        totals["supervised"] += vision_loss_total.item()
        totals["vision"] += vision_loss_total.item()
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
    lambda_corr,
    lambda_depth_delta,
):
    model.eval()

    totals = {
        "total": 0.0,
        "supervised": 0.0,
        "vision": 0.0,
        "correspondence": 0.0,
        "depth_structure": 0.0,
        "depth_delta": 0.0,
        "pose": 0.0,
        "translation_error": 0.0,
        "translation_magnitude_error": 0.0,
        "translation_direction_error": 0.0,
        "yaw_error_deg": 0.0,
        "radius_consistency": 0.0,
        "reprojection": 0.0,
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

        pred_vision_a, pred_vision_b, _, corr_a, corr_b = model(
            img_a,
            img_b,
            compute_pose=False,
            return_corr=True,
        )

        vis_a, *_ = vision_loss(
            pred_vision_a, vision_a,
            occ_thresh=occ_thresh,
            lambda_occ=lambda_occ,
            lambda_radius=lambda_radius,
            lambda_depth=1.0,
        )
        vis_b, *_ = vision_loss(
            pred_vision_b, vision_b,
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

        depth_delta_loss = matched_depth_delta_loss(
            pred_vision_a,
            vision_a,
            pred_vision_b,
            vision_b,
            occ_thresh,
        )

        total_loss = (
            vision_loss_total
            + lambda_corr * corr_loss
            + 0.5 * depth_structure_loss
            + lambda_depth_delta * depth_delta_loss
        )

        radius_a, position_a = relative_cylinder_errors(
            pred_vision_a, vision_a, occ_thresh
        )
        radius_b, position_b = relative_cylinder_errors(
            pred_vision_b, vision_b, occ_thresh
        )

        totals["total"] += total_loss.item()
        totals["supervised"] += vision_loss_total.item()
        totals["vision"] += vision_loss_total.item()
        totals["correspondence"] += corr_loss.item()
        totals["depth_structure"] += depth_structure_loss.item()
        totals["depth_delta"] += depth_delta_loss.item()
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

    # RANSAC is evaluated separately after training in train_v2.ipynb.
    # Keep these keys for compatibility with the existing history/checkpoints.
    metrics["camera_translation_rel_l2"] = float("nan")
    metrics["camera_rotation_rel_l2"] = float("nan")
    metrics["ransac_failure_rate"] = float("nan")

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
    parser.add_argument("--num-workers", type=int, default=4)

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
    parser.add_argument("--lambda-depth-delta", type=float, default=0.25)

    parser.add_argument(
        "--output-dir",
        default="runs/default",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA:", torch.version.cuda)

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

    n_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Model parameters: {n_params / 1e6:.2f} M"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.05,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    history = []
    best_val_loss = float("inf")


    for epoch in range(1, args.epochs + 1):
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
            scaler,
        )

        row = {
            "epoch": epoch,
            "train": train_metrics,
        }

        log = (
            f"Epoch {epoch}: "
            f"tot={train_metrics['total']:.4f} | "
            f"sup={train_metrics['supervised']:.4f} | "
            f"vis={train_metrics['vision']:.4f} | "
            f"corr={train_metrics['correspondence']:.4f} | "
            f"struct={train_metrics['depth_structure']:.4f} | "
            f"depth_delta={train_metrics['depth_delta']:.4f} | "
            f"pose={train_metrics['pose']:.4f} | "
            f"trans={train_metrics['translation_error']:.4f} | "
            f"radius_cons={train_metrics['radius_consistency']:.4f} | "
            f"reproj={train_metrics['reprojection']:.4f}"
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
            )

            row["val"] = val_metrics

            log += (
                f" | val_tot={val_metrics['total']:.4f} | "
                f"val_sup={val_metrics['supervised']:.4f} | "
                f"val_vis={val_metrics['vision']:.4f} | "
                f"val_corr={val_metrics['correspondence']:.4f} | "
                f"val_struct={val_metrics['depth_structure']:.4f} | "
                f"val_depth_delta={val_metrics['depth_delta']:.4f} | "
                f"ransac_fail={100 * val_metrics['ransac_failure_rate']:.1f}% | "
                f"val_pose={val_metrics['pose']:.4f} | "
                f"val_trans={val_metrics['translation_error']:.4f} | "
                f"val_radius_cons={val_metrics['radius_consistency']:.4f} | "
                f"val_reproj={val_metrics['reprojection']:.4f}"
            )
            if val_metrics["total"] < best_val_loss:
                best_val_loss = val_metrics["total"]

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "args": vars(args),
                        "history": history + [row],
                        "best_val_loss": best_val_loss,
                    },
                    os.path.join(
                        args.output_dir,
                        "checkpoints",
                        "best.pt",
                    ),
                )

                log += f" | BEST (val_tot={best_val_loss:.4f})"

        print(log, flush=True)

        history.append(row)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
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
