"""Plot occupancy probabilities and ground truth for validation image pairs.

Place beside dataset.py, augmentations.py, model_pose.py, model.py and
matched_pair_pose.py. Uses the checkpoint format written by train_vision.py.
"""
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_pair(images, targets, predictions, threshold, title, output):
    """Arrays: images HWC, targets/predictions one occupancy value per bin."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), layout="constrained")
    for row, (image, target, prob) in enumerate(zip(images, targets, predictions)):
        target = np.asarray(target)
        prob = np.asarray(prob)
        truth = target >= 0.5
        positive = prob >= threshold
        tp = positive & truth
        fp = positive & ~truth
        fn = ~positive & truth
        precision = tp.sum() / max(positive.sum(), 1)
        recall = tp.sum() / max(truth.sum(), 1)
        f1 = 2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1)
        x = np.arange(len(prob))

        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"Image {'AB'[row]}")
        axes[row, 0].axis("off")
        ax = axes[row, 1]
        ax.step(x, target, where="mid", color="black", alpha=0.65,
                label="Ground truth")
        ax.plot(x, prob, color="tab:blue", label="Predicted probability")
        ax.axhline(threshold, color="gray", linestyle="--",
                   label=f"Threshold {threshold:g}")
        for mask, color, label, marker in (
            (tp, "tab:green", "True positive", "o"),
            (fp, "tab:red", "False positive", "x"),
            (fn, "tab:orange", "False negative", "v"),
        ):
            ax.scatter(x[mask], prob[mask], c=color, marker=marker,
                       s=28, zorder=4, label=label)
        ax.set(xlim=(-0.5, len(prob) - 0.5), ylim=(-0.06, 1.06),
               xlabel="Occupancy bin", ylabel="Occupancy / probability",
               title=(f"P={precision:.3f}  R={recall:.3f}  F1={f1:.3f} | "
                      f"TP={tp.sum()}  FP={fp.sum()}  FN={fn.sum()}"))
        ax.grid(alpha=0.2)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17),
                  ncol=3, fontsize=8)
    fig.suptitle(title)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--indices", type=int, nargs="+",
                        help="Explicit dataset indices; overrides random selection")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="For example cpu or cuda")
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error("--threshold must be between 0 and 1")
    if args.num_samples < 1:
        parser.error("--num-samples must be positive")

    import torch
    from augmentations import build_eval_transform
    from dataset import SceneTwoPairsDataset
    from model_pose import PairImageCylinderModel

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    # Load only your own trusted training checkpoints.
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model_state_dict" not in saved or "args" not in saved:
        parser.error("Expected a train_vision checkpoint with model_state_dict and args")
    cfg = saved["args"]
    keys = ("img_size", "patch_size", "embed_dim", "depth", "num_heads",
            "num_bins", "dropout")
    missing = [key for key in keys if key not in cfg]
    if missing:
        parser.error(f"Checkpoint args missing: {missing}")
    model = PairImageCylinderModel(
        **{key: cfg[key] for key in keys}, in_chans=3,
    )
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.to(device).eval()
    del saved

    dataset = SceneTwoPairsDataset(
        root_dir=os.path.join(args.data_dir, "testdataset"),
        image_size=cfg["img_size"],
        transform=build_eval_transform(cfg["img_size"]),
        debug=False, return_two_pairs=False,
    )
    indices = args.indices
    if indices is None:
        indices = np.random.default_rng(args.seed).choice(
            len(dataset), size=min(args.num_samples, len(dataset)), replace=False,
        ).tolist()
    if any(index < 0 or index >= len(dataset) for index in indices):
        parser.error(f"Indices must be between 0 and {len(dataset) - 1}")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(args.checkpoint).parent.parent / "occupancy_plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for index in indices:
            a, target_a, b, target_b, _ = dataset[index]
            pred_a, pred_b, _ = model(
                a.unsqueeze(0).to(device), b.unsqueeze(0).to(device),
                compute_pose=False, return_corr=False,
            )
            sample = dataset.samples[index]
            output = output_dir / f"occupancy_{index:06d}.png"
            plot_pair(
                [image.permute(1, 2, 0).numpy() for image in (a, b)],
                [target[:, 0].numpy() for target in (target_a, target_b)],
                [pred[0, :, 0].float().cpu().numpy() for pred in (pred_a, pred_b)],
                args.threshold,
                f"Sample {index} | {sample.scene_dir.name} | "
                f"{sample.pair['image1']} / {sample.pair['image2']}",
                output,
            )
            print(output)


if __name__ == "__main__":
    main()
