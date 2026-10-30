"""Pose from a set of explicitly matched cylinder pairs; no RGB inputs."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def matched_tokens(vision_a, vision_b, fov_degrees=90.0, occ_thresh=0.5):
    """Return Kx6 [theta_A, radius_A, depth_A, theta_B, radius_B, depth_B].

    GT channel 3 is used only to pair cylinders; IDs are never input features.
    Only IDs occupying exactly one bin in each view are used. Ambiguous IDs
    are skipped rather than arbitrarily selecting a bin or averaging centers.
    """
    a, b = np.asarray(vision_a, dtype=np.float32), np.asarray(vision_b, dtype=np.float32)
    for v in (a, b):
        if v.ndim != 2 or v.shape[1] < 4 or len(v) < 2:
            raise ValueError('Expected [num_bins, >=4] GT labels')
        occupied = v[:, 0] > occ_thresh
        if not np.isfinite(v[occupied, :4]).all():
            raise ValueError('Nonfinite GT cylinder labels')
        if np.any(v[occupied, 1] < 0) or np.any(v[occupied, 2] <= 0):
            raise ValueError('Occupied bins require radius >= 0 and depth > 0')
        if not np.allclose(v[occupied, 3], np.rint(v[occupied, 3]), atol=1e-4, rtol=0):
            raise ValueError('Cylinder IDs must be integer-valued')
    ia, ib = np.flatnonzero(a[:, 0] > occ_thresh), np.flatnonzero(b[:, 0] > occ_thresh)
    ids_a, ids_b = np.rint(a[ia, 3]).astype(np.int64), np.rint(b[ib, 3]).astype(np.int64)
    rows = []
    ambiguous = 0
    half_fov = math.radians(fov_degrees) / 2
    for cid in np.intersect1d(ids_a, ids_b):
        ma, mb = ia[ids_a == cid], ib[ids_b == cid]
        if len(ma) != 1 or len(mb) != 1:
            ambiguous += 1
            continue
        i, j = int(ma[0]), int(mb[0])
        theta_a = -half_fov + 2 * half_fov * i / (len(a) - 1)
        theta_b = -half_fov + 2 * half_fov * j / (len(b) - 1)
        rows.append([theta_a, a[i, 1], a[i, 2], theta_b, b[j, 1], b[j, 2]])
    return np.asarray(rows, dtype=np.float32).reshape(-1, 6), ambiguous


def has_distinct_centers(tokens, eps=1e-6):
    """At least two noncoincident cylinder centers in each camera view."""
    if len(tokens) < 2:
        return False
    for angle_col, depth_col in ((0, 2), (3, 5)):
        theta, depth = tokens[:, angle_col], tokens[:, depth_col]
        xy = np.stack([depth * np.cos(theta), depth * np.sin(theta)], axis=-1)
        if np.max(np.linalg.norm(xy - xy[0], axis=-1)) <= eps:
            return False
    return True


class MatchedPairPoseModel(nn.Module):
    """Shared token MLP -> self-attention -> masked mean -> pose head.

    No positional encoding: reordering the matched pairs does not change the
    prediction in eval mode, apart from numerical rounding.
    """
    def __init__(self, embed_dim=128, num_heads=4, depth=2, dropout=0.1,
                 fov_degrees=90.0, radius_scale=1.0, distance_scale=10.0):
        super().__init__()
        if min(radius_scale, distance_scale, fov_degrees) <= 0:
            raise ValueError('Input scales and field of view must be positive')
        scale = [math.radians(fov_degrees) / 2, radius_scale, distance_scale] * 2
        self.register_buffer('input_scale', torch.tensor(scale, dtype=torch.float32))
        self.token_mlp = nn.Sequential(
            nn.Linear(6, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=2 * embed_dim,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.pose_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, 4),
        )
        nn.init.normal_(self.pose_head[-1].weight, std=0.001)
        with torch.no_grad():
            self.pose_head[-1].bias.copy_(torch.tensor([1., 0., 0., 1.]))

    def forward(self, pairs, valid):
        # valid: [batch, max_pairs], True only for real cylinder pairs.
        x = self.token_mlp(pairs.float() / self.input_scale)
        x = self.norm(self.blocks(x, src_key_padding_mask=~valid))
        weights = valid.unsqueeze(-1).to(x.dtype)
        pooled = (x * weights).sum(1) / weights.sum(1).clamp_min(1)
        raw = self.pose_head(pooled).float()
        return torch.cat([
            F.normalize(raw[:, :2], dim=-1, eps=1e-6),
            F.normalize(raw[:, 2:], dim=-1, eps=1e-6),
        ], dim=-1)
