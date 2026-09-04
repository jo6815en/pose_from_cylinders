import argparse
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SceneTwoPairsDataset
from model import PairImageCylinderModel
from losses import (
    supervised_loss,
    matched_radius_consistency_loss,
    matched_reprojection_loss_2d,
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
):
    model.train()

    totals = {
        "total": 0.0,
        "supervised": 0.0,
        "vision": 0.0,
        "pose": 0.0,
        "radius_consistency": 0.0,
        "reprojection": 0.0,
    }

    for batch in loader:
        (
            img_a1,
            vision_a1,
            img_b1,
            vision_b1,
            pose_ab1,
            img_a2,
            vision_a2,
            img_b2,
            vision_b2,
            pose_ab2,
        ) = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        pred_vision_a1, pred_vision_b1, pred_pose1 = model(img_a1, img_b1)
        pred_vision_a2, pred_vision_b2, pred_pose2 = model(img_a2, img_b2)

        loss_out1 = supervised_loss(
            pred_vision_a1,
            vision_a1,
            pred_vision_b1,
            vision_b1,
            pred_pose1,
            pose_ab1,
            occ_thresh=occ_thresh,
            lambda_occ=lambda_occ,
            lambda_radius=lambda_radius,
        )

        loss_out2 = supervised_loss(
            pred_vision_a2,
            vision_a2,
            pred_vision_b2,
            vision_b2,
            pred_pose2,
            pose_ab2,
            occ_thresh=occ_thresh,
            lambda_occ=lambda_occ,
            lambda_radius=lambda_radius,
        )

        radius_cons1 = matched_radius_consistency_loss(
            pred_vision_a1,
            vision_a1,
            pred_vision_b1,
            vision_b1,
            relative_pose_pred=pred_pose1,
            matching_mode="gt",
            occ_thresh=occ_thresh,
        )

        radius_cons2 = matched_radius_consistency_loss(
            pred_vision_a2,
            vision_a2,
            pred_vision_b2,
            vision_b2,
            relative_pose_pred=pred_pose2,
            matching_mode="gt",
            occ_thresh=occ_thresh,
        )

        reproj1 = matched_reprojection_loss_2d(
            pred_vision_a1,
            vision_a1,
            pred_vision_b1,
            vision_b1,
            pred_pose1,
            matching_mode="gt",
            occ_thresh=occ_thresh,
            fov_degrees=90.0,
        )

        reproj2 = matched_reprojection_loss_2d(
            pred_vision_a2,
            vision_a2,
            pred_vision_b2,
            vision_b2,
            pred_pose2,
            matching_mode="gt",
            occ_thresh=occ_thresh,
            fov_degrees=90.0,
        )

        sup_loss = (loss_out1["total"] + loss_out2["total"]) / 2.0
        radius_cons = (radius_cons1 + radius_cons2) / 2.0
        reproj = (reproj1 + reproj2) / 2.0

        # Same weighting as the current notebook.
        loss = sup_loss # + 5.0 * radius_cons + 10.0 * reproj

        loss.backward()
        optimizer.step()

        totals["total"] += loss.item()
        totals["supervised"] += sup_loss.item()
        totals["vision"] += (
            (loss_out1["vision_a"] + loss_out2["vision_a"]) / 2.0
        ).item()
        totals["pose"] += (
            (loss_out1["pose"] + loss_out2["pose"]) / 2.0
        ).item()
        totals["radius_consistency"] += radius_cons.item()
        totals["reprojection"] += reproj.item()

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
):
    model.eval()

    totals = {
        "total": 0.0,
        "supervised": 0.0,
        "vision": 0.0,
        "pose": 0.0,
        "radius_consistency": 0.0,
        "reprojection": 0.0,
    }

    for batch in loader:
        (
            img_a,
            vision_a,
            img_b,
            vision_b,
            pose_ab,
        ) = move_batch_to_device(batch, device)

        pred_vision_a, pred_vision_b, pred_pose = model(img_a, img_b)

        loss_out = supervised_loss(
            pred_vision_a,
            vision_a,
            pred_vision_b,
            vision_b,
            pred_pose,
            pose_ab,
            occ_thresh=occ_thresh,
            lambda_occ=lambda_occ,
            lambda_radius=lambda_radius,
        )

        radius_cons = matched_radius_consistency_loss(
            pred_vision_a,
            vision_a,
            pred_vision_b,
            vision_b,
            relative_pose_pred=pred_pose,
            matching_mode="gt",
            occ_thresh=occ_thresh,
        )

        reproj = matched_reprojection_loss_2d(
            pred_vision_a,
            vision_a,
            pred_vision_b,
            vision_b,
            pred_pose,
            matching_mode="gt",
            occ_thresh=occ_thresh,
            fov_degrees=90.0,
        )

        totals["total"] += loss_out["total"].item()
        totals["supervised"] += loss_out["total"].item()
        totals["vision"] += loss_out["vision_a"].item()
        totals["pose"] += loss_out["pose"].item()
        totals["radius_consistency"] += radius_cons.item()
        totals["reprojection"] += reproj.item()

    return {
        key: value / len(loader)
        for key, value in totals.items()
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        default="/nobackup/proj/disk/midlevel_representations/personal/johanna/data",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-interval", type=int, default=10)

    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--num-bins", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--lambda-occ", type=float, default=10.0)
    parser.add_argument("--lambda-radius", type=float, default=10.0)
    parser.add_argument("--occ-thresh", type=float, default=0.5)

    parser.add_argument(
        "--output-dir",
        default="runs/default",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    )

    val_dataset = SceneTwoPairsDataset(
        root_dir=os.path.join(args.data_dir, "valdataset"),
        image_size=args.img_size,
        debug=False,
        return_two_pairs=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
    )

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
            f"pose={train_metrics['pose']:.4f} | "
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
            )

            row["val"] = val_metrics

            log += (
                f" | val_tot={val_metrics['total']:.4f} | "
                f"val_sup={val_metrics['supervised']:.4f} | "
                f"val_vis={val_metrics['vision']:.4f} | "
                f"val_pose={val_metrics['pose']:.4f} | "
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
