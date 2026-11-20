"""Two-stage vision training with unlabeled forest consistency regularization.

The supervised training and validation paths follow train_vision.py. Each
forest image is also processed twice with independent color-only augmentation;
the consistency loss compares corresponding angular bins, so the input
geometry and bin alignment are unchanged.
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

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


class ForestConsistencyDataset(Dataset):
    """Unlabeled forest images with two independent photometric views."""

    def __init__(self, dataset_dir, image_size):
        self.root = Path(dataset_dir)
        self.image_dir = self.root / "images"
        with open(self.root / "relative_poses.json", "r", encoding="utf-8") as f:
            pairs = json.load(f)
        # Each pair contributes its two images. Duplicates are harmless and
        # retain the complete image set used by the pair dataset.
        self.image_names = [name for pair in pairs for name in
                            (pair["from_image"], pair["to_image"])]
        self.base_transform = build_eval_transform(image_size)
        self.photo = transforms.ColorJitter(
            brightness=0.25, contrast=0.25, saturation=0.20, hue=0.04
        )
        self.gray = transforms.RandomGrayscale(p=0.08)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        path = self.image_dir / self.image_names[idx]
        with Image.open(path) as im:
            image = im.convert("RGB")
        # ColorJitter/Grayscale alter appearance only; the same deterministic
        # resize/crop/normalization is applied to both views.
        view_a = self.base_transform(self.gray(self.photo(image.copy())))
        view_b = self.base_transform(self.gray(self.photo(image.copy())))
        return view_a, view_b


def move_batch_to_device(batch, device):
    return tuple(x.to(device, non_blocking=True) for x in batch)


def consistency_loss(pred_a, pred_b, occ_thresh=0.5, occupancy_only=False):
    """Return total consistency and its occupancy/geometry components."""
    with torch.autocast(device_type=pred_a.device.type, enabled=False):
        a, b = pred_a.float(), pred_b.float()
        occ = F.mse_loss(a[..., 0], b[..., 0])
        if occupancy_only:
            return occ, occ, occ.new_zeros(()), occ.new_zeros(())
        # Avoid forcing arbitrary radius/depth/width values in empty bins.
        mask = ((a[..., 0].detach() > occ_thresh) |
                (b[..., 0].detach() > occ_thresh)).float()
        geometry = F.smooth_l1_loss(a[..., 1:], b[..., 1:], reduction="none")
        active_fraction = mask.mean()
        geometry = (geometry * mask.unsqueeze(-1)).sum() / (
            mask.sum().clamp_min(1.0) * geometry.shape[-1]
        )
        return occ + geometry, occ, geometry, active_fraction


def occupancy_entropy(pred_a, pred_b):
    """Mean Bernoulli entropy of the two forest-view occupancy predictions.

    Minimizing this term favors confident probabilities near 0 or 1. The
    batch-level occupancy prior remains active to constrain their mean rate.
    """
    with torch.autocast(device_type=pred_a.device.type, enabled=False):
        eps = 1e-6
        p = torch.cat((pred_a[..., 0].float().reshape(-1),
                       pred_b[..., 0].float().reshape(-1)))
        p = p.clamp(eps, 1.0 - eps)
        return -(p * p.log() + (1.0 - p) * torch.log1p(-p)).mean()


def topk_agreement_feedback(pred_a, pred_b, topk_fraction=0.072):
    """Give weak positive pseudo-feedback to bins ranked highly in both views.

    The top-k selection is detached from autograd. Only bins selected by both
    augmentations receive positive targets; all other bins are left unlabeled
    by this term. The occupancy-rate prior controls the overall prediction rate.
    """
    with torch.autocast(device_type=pred_a.device.type, enabled=False):
        p_a = pred_a[..., 0].float()
        p_b = pred_b[..., 0].float()
        batch_size, num_bins = p_a.shape
        k = max(1, min(num_bins, int(math.ceil(topk_fraction * num_bins))))

        idx_a = torch.topk(p_a.detach(), k=k, dim=1).indices
        idx_b = torch.topk(p_b.detach(), k=k, dim=1).indices
        mask_a = torch.zeros_like(p_a, dtype=torch.bool).scatter_(1, idx_a, True)
        mask_b = torch.zeros_like(p_b, dtype=torch.bool).scatter_(1, idx_b, True)
        mutual_mask = mask_a & mask_b
        mutual_count = mutual_mask.sum(dim=1).float()
        overlap_fraction = (mutual_count / float(k)).mean()

        if mutual_mask.any():
            selected_a = p_a[mutual_mask].clamp(1e-6, 1.0 - 1e-6)
            selected_b = p_b[mutual_mask].clamp(1e-6, 1.0 - 1e-6)
            loss_a = F.binary_cross_entropy(selected_a, torch.ones_like(selected_a))
            loss_b = F.binary_cross_entropy(selected_b, torch.ones_like(selected_b))
            feedback_loss = 0.5 * (loss_a + loss_b)
        else:
            feedback_loss = (p_a.sum() + p_b.sum()) * 0.0

        return feedback_loss, mutual_count.mean(), overlap_fraction


def train_one_epoch(model, loader, forest_loader, optimizer, device, args, scaler,
                    epoch, consistency_weight, occupancy_prior_weight,
                    entropy_weight, topk_feedback_weight):
    model.train()
    totals = {k: 0.0 for k in ("total", "vision", "vision_occupancy", "vision_radius",
                               "vision_depth", "vision_width", "correspondence",
                               "depth_structure", "depth_delta", "consistency",
                               "consistency_occupancy", "consistency_geometry",
                               "occupancy_prior", "occupancy_entropy",
                               "topk_feedback", "topk_mutual_bins",
                               "topk_overlap_fraction",
                               "supervised_occ_target",
                               "forest_occ_mean", "forest_occ_active_fraction",
                               "forest_geometry_active_fraction")}
    forest_iter = iter(forest_loader)
    for batch in loader:
        img_a, vision_a, img_b, vision_b, _pose_ab = move_batch_to_device(batch, device)
        try:
            forest_a, forest_b = next(forest_iter)
        except StopIteration:
            forest_iter = iter(forest_loader)
            forest_a, forest_b = next(forest_iter)
        forest_a, forest_b = forest_a.to(device, non_blocking=True), forest_b.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            out = model(img_a, img_b, return_corr=not args.occupancy_only, compute_pose=False)
            pred_a, pred_b = out[:2]
            if args.occupancy_only:
                with torch.autocast(device_type=device.type, enabled=False):
                    occ_a = F.binary_cross_entropy(pred_a[..., 0].float().clamp(1e-6, 1-1e-6), vision_a[..., 0].float())
                    occ_b = F.binary_cross_entropy(pred_b[..., 0].float().clamp(1e-6, 1-1e-6), vision_b[..., 0].float())
                vis_loss = args.lambda_occ * 0.5 * (occ_a + occ_b)
                rad_a = rad_b = dep_a = dep_b = width_a = width_b = vis_loss.new_zeros(())
                corr = struct = delta = vis_loss.new_zeros(())
            else:
                va, occ_a, rad_a, dep_a, width_a = vision_loss(
                    pred_a, vision_a, args.occ_thresh, args.lambda_occ, args.lambda_radius,
                    args.lambda_depth, args.lambda_width)
                vb, occ_b, rad_b, dep_b, width_b = vision_loss(
                    pred_b, vision_b, args.occ_thresh, args.lambda_occ, args.lambda_radius,
                    args.lambda_depth, args.lambda_width)
                vis_loss = 0.5 * (va + vb)
                corr_a, corr_b = out[3:]
                corr = patch_correspondence_loss(corr_a, vision_a, corr_b, vision_b,
                                                 temperature=args.temperature)
                struct = 0.5 * (relative_depth_structure_loss(pred_a, vision_a, args.occ_thresh)
                                + relative_depth_structure_loss(pred_b, vision_b, args.occ_thresh))
                delta = matched_depth_delta_loss(pred_a, vision_a, pred_b, vision_b, args.occ_thresh)

            forest_out = model(forest_a, forest_b, return_corr=False, compute_pose=False)
            pred_fa, pred_fb = forest_out[:2]
            consistency, consistency_occ, consistency_geom, geom_active = consistency_loss(
                pred_fa, pred_fb, args.forest_occ,
                occupancy_only=args.occupancy_only)
            # Match only the batch-level occupancy rate, not bin locations.
            # The GT rate is a target statistic; it receives no gradient.
            forest_occ_mean = 0.5 * (pred_fa[..., 0].float().mean()
                                     + pred_fb[..., 0].float().mean())
            supervised_occ_target = 0.5 * (
                vision_a[..., 0].float().mean() + vision_b[..., 0].float().mean()
            )
            # Occupancy predictions are probabilities (already passed through
            # sigmoid), so use BCE in float32 with autocast disabled.
            with torch.autocast(device_type=device.type, enabled=False):
                occupancy_prior = F.binary_cross_entropy(
                    forest_occ_mean.float().clamp(1e-6, 1.0 - 1e-6),
                    supervised_occ_target.float().clamp(1e-6, 1.0 - 1e-6),
                )
            entropy = occupancy_entropy(pred_fa, pred_fb)
            topk_feedback, topk_mutual_bins, topk_overlap_fraction = (
                topk_agreement_feedback(pred_fa, pred_fb, args.topk_fraction)
            )
            loss = vis_loss + args.lambda_corr * corr + args.lambda_depth_struct * struct
            loss = (loss + args.lambda_depth_delta * delta
                    + consistency_weight * consistency
                    + occupancy_prior_weight * occupancy_prior
                    + entropy_weight * entropy
                    + topk_feedback_weight * topk_feedback)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        values = {"total": loss, "vision": vis_loss,
                  "vision_occupancy": 0.5 * (occ_a + occ_b),
                  "vision_radius": 0.5 * (rad_a + rad_b),
                  "vision_depth": 0.5 * (dep_a + dep_b),
                  "vision_width": 0.5 * (width_a + width_b),
                  "correspondence": corr, "depth_structure": struct,
                  "depth_delta": delta, "consistency": consistency,
                  "consistency_occupancy": consistency_occ,
                  "consistency_geometry": consistency_geom,
                  "occupancy_prior": occupancy_prior,
                  "occupancy_entropy": entropy,
                  "topk_feedback": topk_feedback,
                  "topk_mutual_bins": topk_mutual_bins,
                  "topk_overlap_fraction": topk_overlap_fraction,
                  "supervised_occ_target": supervised_occ_target,
                  "forest_occ_mean": forest_occ_mean,
                  "forest_occ_active_fraction": 0.5 * (
                      (pred_fa[..., 0] > args.forest_occ).float().mean()
                      + (pred_fb[..., 0] > args.forest_occ).float().mean()),
                  "forest_geometry_active_fraction": geom_active}
        for key, value in values.items():
            totals[key] += value.detach().item()
    return {key: value / len(loader) for key, value in totals.items()}


# Validation remains the supervised validation used by train_vision.py.
@torch.no_grad()
def validate(model, loader, device, args):
    model.eval()
    sums = {k: 0.0 for k in ("total", "vision", "vision_occupancy", "vision_radius",
                             "vision_depth", "vision_width", "correspondence",
                             "depth_structure", "depth_delta")}
    n = 0
    for batch in loader:
        img_a, vision_a, img_b, vision_b, _ = move_batch_to_device(batch, device)
        out = model(img_a, img_b, return_corr=not args.occupancy_only, compute_pose=False)
        pa, pb = out[:2]
        if args.occupancy_only:
            occ_a = F.binary_cross_entropy(pa[..., 0].float().clamp(1e-6, 1-1e-6), vision_a[..., 0].float())
            occ_b = F.binary_cross_entropy(pb[..., 0].float().clamp(1e-6, 1-1e-6), vision_b[..., 0].float())
            vis = args.lambda_occ * 0.5 * (occ_a + occ_b)
            ra = rb = da = db = wa = wb = vis.new_zeros(())
            corr = struct = delta = vis.new_zeros(())
        else:
            va, occ_a, ra, da, wa = vision_loss(pa, vision_a, args.occ_thresh, args.lambda_occ,
                                                args.lambda_radius, args.lambda_depth, args.lambda_width)
            vb, occ_b, rb, db, wb = vision_loss(pb, vision_b, args.occ_thresh, args.lambda_occ,
                                                args.lambda_radius, args.lambda_depth, args.lambda_width)
            vis = 0.5 * (va + vb)
            corr = patch_correspondence_loss(out[3], vision_a, out[4], vision_b, temperature=args.temperature)
            struct = 0.5 * (relative_depth_structure_loss(pa, vision_a, args.occ_thresh)
                            + relative_depth_structure_loss(pb, vision_b, args.occ_thresh))
            delta = matched_depth_delta_loss(pa, vision_a, pb, vision_b, args.occ_thresh)
        vals = {"total": vis + args.lambda_corr*corr + args.lambda_depth_struct*struct + args.lambda_depth_delta*delta,
                "vision": vis, "vision_occupancy": 0.5*(occ_a+occ_b), "vision_radius": 0.5*(ra+rb),
                "vision_depth": 0.5*(da+db), "vision_width": 0.5*(wa+wb), "correspondence": corr,
                "depth_structure": struct, "depth_delta": delta}
        for key, value in vals.items(): sums[key] += value.item()
        n += 1
    if not n:
        raise RuntimeError("Validation loader is empty.")
    return {k: v/n for k, v in sums.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/nobackup/proj/disk/midlevel_representations/personal/johanna/data")
    parser.add_argument("--forest-dir", default="forest_torpa", help="Folder containing images/ and relative_poses.json")
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
    parser.add_argument("--forest-occ", type=float, default=0.3)
    parser.add_argument("--lambda-occ", type=float, default=10.0)
    parser.add_argument("--lambda-radius", type=float, default=10.0)
    parser.add_argument("--lambda-depth", type=float, default=1.0)
    parser.add_argument("--lambda-width", type=float, default=0.1)
    parser.add_argument("--lambda-corr", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--lambda-depth-delta", type=float, default=0.35)
    parser.add_argument("--lambda-depth-struct", type=float, default=0.5)
    parser.add_argument("--lambda-consistency", type=float, default=1.0)
    parser.add_argument("--consistency-warmup-epochs", type=int, default=5,
                        help="Epochs before consistency starts increasing")
    parser.add_argument("--consistency-ramp-epochs", type=int, default=10,
                        help="Linear ramp duration to --lambda-consistency")
    parser.add_argument("--lambda-occ-prior", type=float, default=0.5,
                        help="Initial weight for matching forest mean occupancy to the GT batch rate")
    parser.add_argument("--occ-prior-final-factor", type=float, default=0.1,
                        help="Final occupancy-prior weight as a fraction of its initial weight")
    parser.add_argument("--lambda-entropy", type=float, default=0.01,
                        help="Maximum weight for minimizing forest occupancy entropy; 0 disables it")
    parser.add_argument("--topk-fraction", type=float, default=0.072,
                        help="Per-view occupancy fraction used for top-k forest pseudo-feedback")
    parser.add_argument("--lambda-topk-feedback", type=float, default=0.05,
                        help="Maximum weight for positive feedback on bins in both views' top-k")
    parser.add_argument("--output-dir", default="runs/vision_semi")
    parser.add_argument("--stage", choices=("full", "occupancy", "geometry"), default="full")
    parser.add_argument("--init-checkpoint", default=None)
    args = parser.parse_args()
    args.occupancy_only = args.stage == "occupancy"
    
    if args.stage == "geometry" and not args.init_checkpoint:
        parser.error("--stage geometry requires --init-checkpoint")
    if args.init_checkpoint and args.stage != "geometry":
        parser.error("--init-checkpoint is only used with --stage geometry")
    if min(args.epochs, args.batch_size, args.val_interval, args.lr, args.lambda_occ) <= 0:
        parser.error("Epochs, batch size, validation interval, learning rate and occupancy weight must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    if args.consistency_warmup_epochs < 0 or args.consistency_ramp_epochs < 1:
        parser.error("Consistency warmup must be nonnegative and ramp duration must be positive")
    if (args.lambda_consistency < 0 or args.lambda_occ_prior < 0
            or args.lambda_entropy < 0 or args.lambda_topk_feedback < 0):
        parser.error("Consistency, prior, entropy, and top-k weights must be nonnegative")
    if not 0.0 < args.topk_fraction <= 1.0:
        parser.error("--topk-fraction must be in (0, 1]")
    if not 0.0 <= args.occ_prior_final_factor <= 1.0:
        parser.error("--occ-prior-final-factor must be between 0 and 1")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda": torch.backends.cudnn.benchmark = True
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    train_dataset = SceneTwoPairsDataset(root_dir=os.path.join(args.data_dir, "valdataset"),
        image_size=args.img_size, debug=False, return_two_pairs=False,
        transform=build_train_transform(args.img_size))
    val_dataset = SceneTwoPairsDataset(root_dir=os.path.join(args.data_dir, "testdataset"),
        image_size=args.img_size, debug=False, return_two_pairs=False,
        transform=build_eval_transform(args.img_size))
    forest_dataset = ForestConsistencyDataset(args.forest_dir, args.img_size)
    
    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                         pin_memory=(device.type == "cuda"), persistent_workers=(args.num_workers > 0))
    
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    forest_loader = DataLoader(forest_dataset, shuffle=True, drop_last=False, **loader_kwargs)
    
    if not len(forest_dataset): parser.error("Forest dataset contains no image pairs")
    print(f"Device: {device} | supervised={len(train_dataset)} | validation={len(val_dataset)} | forest={len(forest_dataset)}")

    model = PairImageCylinderModel(img_size=args.img_size, patch_size=args.patch_size, in_chans=3,
        embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads,
        num_bins=args.num_bins, dropout=args.dropout).to(device)
    
    if args.init_checkpoint:
        saved = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        model.load_vision_checkpoint(args.init_checkpoint)
        print("Loaded checkpoint", args.init_checkpoint, "epoch", saved.get("epoch"))
    
    model.backbone.requires_grad_(True); model.vision_head.requires_grad_(True)
    model.pose_head.requires_grad_(False); model.corr_decoder.requires_grad_(not args.occupancy_only)
    
    if args.occupancy_only:
        for name in ("radius_head", "depth_head", "width_head"):
            getattr(model.vision_head, name).requires_grad_(False)
    
    raw_model = model
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    
    best = float("inf")
    best_epoch = None
    history = {"epoch": [], "loss": [], "consistency_loss": [],
               "consistency_occupancy": [], "consistency_geometry": [],
               "consistency_weight": [], "occupancy_prior_loss": [],
               "occupancy_prior_weight": [], "occupancy_entropy": [],
               "entropy_weight": [], "topk_feedback_loss": [],
               "topk_feedback_weight": [], "topk_mutual_bins": [],
               "topk_overlap_fraction": [], "supervised_occ_target": [],
               "forest_occ_mean": [], "forest_occ_active_fraction": [],
               "forest_geometry_active_fraction": [],
               "val_total": [], "val_epoch": []}
    
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        # Keep consistency weak early, then ramp it to its configured maximum.
        consistency_progress = min(
            1.0,
            max(0.0, (epoch - args.consistency_warmup_epochs)
                / args.consistency_ramp_epochs),
        )
        consistency_weight = args.lambda_consistency * consistency_progress
        # Start with a stronger occupancy-rate anchor and anneal it gradually.
        prior_progress = (epoch - 1) / max(args.epochs - 1, 1)
        occupancy_prior_weight = args.lambda_occ_prior * (
            1.0 - prior_progress * (1.0 - args.occ_prior_final_factor)
        )
        # Use the same warmup/ramp as consistency to avoid sharpening early
        # predictions before the supervised initialization has adapted.
        entropy_weight = args.lambda_entropy * consistency_progress
        topk_feedback_weight = args.lambda_topk_feedback * consistency_progress
        tr = train_one_epoch(model, train_loader, forest_loader, optimizer, device,
                             args, scaler, epoch, consistency_weight,
                             occupancy_prior_weight, entropy_weight,
                             topk_feedback_weight)
        history["epoch"].append(epoch); history["loss"].append(tr["total"])
        history["consistency_loss"].append(tr["consistency"])
        history["consistency_weight"].append(consistency_weight)
        history["occupancy_prior_loss"].append(tr["occupancy_prior"])
        history["occupancy_prior_weight"].append(occupancy_prior_weight)
        history["occupancy_entropy"].append(tr["occupancy_entropy"])
        history["entropy_weight"].append(entropy_weight)
        history["topk_feedback_loss"].append(tr["topk_feedback"])
        history["topk_feedback_weight"].append(topk_feedback_weight)
        history["topk_mutual_bins"].append(tr["topk_mutual_bins"])
        history["topk_overlap_fraction"].append(tr["topk_overlap_fraction"])
        history["supervised_occ_target"].append(tr["supervised_occ_target"])
        for key in ("consistency_occupancy", "consistency_geometry", "forest_occ_mean",
                    "forest_occ_active_fraction", "forest_geometry_active_fraction"):
            history[key].append(tr[key])
        log = (f"Epoch {epoch}: total={tr['total']:.4f} supervised={tr['vision']:.4f} "
               f"cons={tr['consistency']:.4f} "
               f"(w={consistency_weight:.3f}) "
               f"(occ={tr['consistency_occupancy']:.4f}, geom={tr['consistency_geometry']:.4f}) "
               f"occ_prior={tr['occupancy_prior']:.4f}*{occupancy_prior_weight:.3f} "
               f"entropy={tr['occupancy_entropy']:.4f}*{entropy_weight:.4f} "
               f"topk={tr['topk_feedback']:.4f}*{topk_feedback_weight:.4f} "
               f"topk_mutual={tr['topk_mutual_bins']:.2f}/{math.ceil(args.topk_fraction * args.num_bins)} "
               f"topk_overlap={tr['topk_overlap_fraction']:.3f} "
               f"gt_occ={tr['supervised_occ_target']:.3f} "
               f"forest_occ_mean={tr['forest_occ_mean']:.3f} "
               f"forest_occ>{args.forest_occ:g}={tr['forest_occ_active_fraction']:.3f} "
               f"geom_active={tr['forest_geometry_active_fraction']:.3f}")
        
        if epoch == 1 or epoch % args.val_interval == 0:
            val = validate(model, val_loader, device, args)
            history["val_epoch"].append(epoch); history["val_total"].append(val["total"])
            log += f" | val_total={val['total']:.4f}"
            if val["total"] < best:
                best = val["total"]
                best_epoch = epoch
                torch.save({"epoch": epoch, "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(), "args": vars(args),
                    "training_stage": "vision_semi_" + args.stage, "history": history,
                    "best_val_loss": best}, os.path.join(args.output_dir, "checkpoints", "best.pt"))
                log += " | BEST"
        
        log += f" | time={time.perf_counter()-start:.1f}s"
        print(log, flush=True)
        
        torch.save({"epoch": epoch, "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "args": vars(args),
            "training_stage": "vision_semi_" + args.stage, "history": history},
            os.path.join(args.output_dir, "checkpoints", "latest.pt"))
        with open(os.path.join(args.output_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print("\nTraining summary")
    print(f"  Best validation loss: {best:.4f} (epoch {best_epoch})")
    print(f"  Final training loss: {history['loss'][-1]:.4f}")
    print(f"  Final consistency: {history['consistency_loss'][-1]:.6f} "
          f"(occupancy={history['consistency_occupancy'][-1]:.6f}, "
          f"geometry={history['consistency_geometry'][-1]:.6f})")
    print(f"  Final consistency weight: {history['consistency_weight'][-1]:.4f}")
    print(f"  Final occupancy prior: loss={history['occupancy_prior_loss'][-1]:.6f}, "
          f"weight={history['occupancy_prior_weight'][-1]:.4f}, "
          f"GT occupancy target={history['supervised_occ_target'][-1]:.4f}")
    print(f"  Final occupancy entropy: loss={history['occupancy_entropy'][-1]:.6f}, "
          f"weight={history['entropy_weight'][-1]:.4f}")
    print(f"  Final top-k feedback: loss={history['topk_feedback_loss'][-1]:.6f}, "
          f"weight={history['topk_feedback_weight'][-1]:.4f}, "
          f"mutual bins={history['topk_mutual_bins'][-1]:.3f}, "
          f"overlap={history['topk_overlap_fraction'][-1]:.4f}")
    print(f"  Forest mean occupancy: {history['forest_occ_mean'][-1]:.4f}")
    print(f"  Forest fraction above occupancy threshold ({args.forest_occ:g}): "
          f"{history['forest_occ_active_fraction'][-1]:.4f}")
    print(f"  Forest fraction activating geometry consistency: "
          f"{history['forest_geometry_active_fraction'][-1]:.4f}")
    print(f"  Per-epoch metrics saved to: {os.path.join(args.output_dir, 'history.json')}")


if __name__ == "__main__":
    main()
