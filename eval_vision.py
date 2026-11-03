"""Evaluate train_vision.py checkpoints with the existing metrics.evaluate_vision."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import SceneTwoPairsDataset
from metrics import evaluate_vision
from model_pose import PairImageCylinderModel


class VisionOnlyModel(PairImageCylinderModel):
    # Also supports an older evaluate_vision that calls model(img_a, img_b).
    def forward(self, img_a, img_b, **kwargs):
        kwargs["compute_pose"] = False
        return super().forward(img_a, img_b, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--data-dir",
        default="/nobackup/proj/disk/midlevel_representations/personal/johanna/data",
    )
    parser.add_argument("--split", default="testdataset")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--occ-threshold", type=float, default=0.20)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("Require --batch-size > 0 and --num-workers >= 0")
    if not 0 <= args.occ_threshold <= 1:
        parser.error("Require 0 <= --occ-threshold <= 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("args", {})
    model = VisionOnlyModel(
        img_size=cfg.get("img_size", 128),
        patch_size=tuple(cfg.get("patch_size", (16, 4))),
        in_chans=3,
        embed_dim=cfg.get("embed_dim", 192),
        depth=cfg.get("depth", 4),
        num_heads=cfg.get("num_heads", 4),
        num_bins=cfg.get("num_bins", 128),
        dropout=cfg.get("dropout", 0.1),
        fov_degrees=cfg.get("fov_degrees", 90.0),
    )
    # Load only the vision modules; no trained pose checkpoint is needed.
    model.load_vision_checkpoint(args.checkpoint)
    model.to(device).eval()

    # Same deterministic preprocessing as the inference notebook.
    dataset = SceneTwoPairsDataset(
        root_dir=Path(args.data_dir) / args.split,
        image_size=cfg.get("img_size", 128),
        return_two_pairs=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    print("Device:", device)
    print("Checkpoint:", args.checkpoint)
    print("Epoch:", checkpoint.get("epoch"), "Stage:", checkpoint.get("training_stage"))
    print("Dataset:", Path(args.data_dir) / args.split)
    if cfg.get("stage") == "occupancy" or checkpoint.get("training_stage") == "vision_occupancy":
        print("Occupancy-only checkpoint: radius and depth heads have not been trained.")
    return evaluate_vision(model, loader, device, occ_threshold=args.occ_threshold)


if __name__ == "__main__":
    main()
