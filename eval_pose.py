"""Evaluate a joint model_pose checkpoint with the existing metrics.evaluate_pose."""

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import SceneTwoPairsDataset
from metrics import evaluate_pose
from model_pose import PairImageCylinderModel


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
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("Require --batch-size > 0 and --num-workers >= 0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("args", {})
    state = checkpoint["model_state_dict"]
    if "pose_head.token_mlp.0.weight" not in state:
        parser.error("Select a checkpoint from the joint model_pose (e.g. train_tot.py)")

    scale = state["pose_head.input_scale"]
    pose_depth = len({
        key.split(".")[3]
        for key in state if key.startswith("pose_head.blocks.layers.")
    })
    model = PairImageCylinderModel(
        img_size=cfg.get("img_size", 128),
        patch_size=tuple(cfg.get("patch_size", (16, 4))),
        in_chans=3,
        embed_dim=cfg.get("embed_dim", 192),
        depth=cfg.get("depth", 4),
        num_heads=cfg.get("num_heads", 4),
        num_bins=cfg.get("num_bins", 128),
        dropout=cfg.get("dropout", 0.1),
        pose_embed_dim=state["pose_head.token_mlp.0.weight"].shape[0],
        pose_num_heads=cfg.get("pose_num_heads", 4),
        pose_depth=pose_depth,
        pose_dropout=cfg.get("pose_dropout", 0.1),
        fov_degrees=math.degrees(float(scale[0])) * 2,
        radius_scale=float(scale[1]),
        distance_scale=float(scale[2]),
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    # Same deterministic preprocessing as the inference cells in train_tot.ipynb.
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
    print("Matching: GT bins and GT IDs; radius and depth are predicted.")
    return evaluate_pose(model, loader, device)


if __name__ == "__main__":
    main()
