import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from matched_pair_pose import MatchedPairPoseModel

class PatchEmbed(nn.Module):
    def __init__(self, img_size=128, patch_size=(16, 8), in_chans=3, embed_dim=192):
        super().__init__()

        if isinstance(patch_size, int):
            patch_h = patch_w = patch_size
        else:
            patch_h, patch_w = patch_size

        assert img_size % patch_h == 0
        assert img_size % patch_w == 0

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.grid_h = img_size // patch_h
        self.grid_w = img_size // patch_w
        self.num_patches = self.grid_h * self.grid_w

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=(patch_h, patch_w),
            stride=(patch_h, patch_w),
        )

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=3.0, dropout=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class PairBlock(nn.Module):
    """Lightweight V2-style block: per-view self-attention, bidirectional cross-attention, MLP."""
    def __init__(self, dim=192, num_heads=4, mlp_ratio=3.0, dropout=0.0):
        super().__init__()
        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_cross_q = nn.LayerNorm(dim)
        self.norm_cross_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, dropout)

    def forward(self, a, b, return_attention=False):
        an, bn = self.norm_self(a), self.norm_self(b)

        da, _ = self.self_attn(an, an, an, need_weights=False)
        db, _ = self.self_attn(bn, bn, bn, need_weights=False)

        a, b = a + da, b + db

        aq, bq = self.norm_cross_q(a), self.norm_cross_q(b)
        akv, bkv = self.norm_cross_kv(a), self.norm_cross_kv(b)

        da, attn_ab = self.cross_attn(
            aq, bkv, bkv, need_weights=return_attention, average_attn_weights=False
        )
        db, attn_ba = self.cross_attn(
            bq, akv, akv, need_weights=return_attention, average_attn_weights=False
        )

        a, b = a + da, b + db

        a = a + self.mlp(self.norm_mlp(a))
        b = b + self.mlp(self.norm_mlp(b))

        if return_attention:
            return a, b, attn_ab, attn_ba

        return a, b

class PairViTBackbone(nn.Module):
    def __init__(self, img_size=128, patch_size=(16,8), in_chans=3, embed_dim=192,
                 depth=4, num_heads=4, dropout=0.0, num_register_tokens=2, mlp_ratio=3.0):
        super().__init__()
        self.num_register_tokens = num_register_tokens
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        n = self.patch_embed.num_patches
        self.camera_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
        self.patch_pos_embed = nn.Parameter(torch.zeros(1, n, embed_dim))
        self.view_embed = nn.Parameter(torch.zeros(1, 2, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            PairBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        for p in (self.camera_token, self.register_tokens, self.patch_pos_embed, self.view_embed):
            nn.init.trunc_normal_(p, std=0.02)

    def _tokens(self, img, view_idx):
        B = img.shape[0]
        v = self.view_embed[:, view_idx:view_idx + 1]
        patches = self.patch_embed(img) + self.patch_pos_embed + v
        cam = self.camera_token.expand(B, -1, -1) + v
        regs = self.register_tokens.expand(B, -1, -1) + v
        return self.pos_drop(torch.cat([cam, regs, patches], dim=1))

    def forward(self, img_a, img_b, return_attention=False):
        a = self._tokens(img_a, 0)
        b = self._tokens(img_b, 1)

        attn_ab_all = []
        attn_ba_all = []

        for block in self.blocks:
            if return_attention:
                a, b, attn_ab, attn_ba = block(a, b, return_attention=True)
                attn_ab_all.append(attn_ab)
                attn_ba_all.append(attn_ba)
            else:
                a, b = block(a, b)

        a = self.norm(a)
        b = self.norm(b)

        start = 1 + self.num_register_tokens
        outputs = (a[:, 0], b[:, 0], a[:, start:], b[:, start:])

        if return_attention:
            return (*outputs, attn_ab_all, attn_ba_all)

        return outputs

class CylinderDecoderBlock(nn.Module):
    def __init__(self, embed_dim=192, num_heads=4, mlp_ratio=3.0, dropout=0.0):
        super().__init__()

        self.norm_self = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_cross_q = nn.LayerNorm(embed_dim)
        self.norm_cross_kv = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_mlp = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, q, patches):
        qn = self.norm_self(q)
        dq, _ = self.self_attn(qn, qn, qn, need_weights=False)
        q = q + dq

        qn = self.norm_cross_q(q)
        kv = self.norm_cross_kv(patches)

        dq, _ = self.cross_attn(qn, kv, kv, need_weights=False)
        q = q + dq

        q = q + self.mlp(self.norm_mlp(q))

        return q

class CylinderDecoder(nn.Module):
    def __init__(
        self,
        embed_dim=192,
        num_heads=4,
        num_bins=128,
        grid_h=8,
        grid_w=16,
        mlp_ratio=3.0,
        dropout=0.0,
        depth=3,
        fov_degrees=90.0,
    ):
        super().__init__()

        self.num_bins = num_bins
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.fov_degrees = fov_degrees

        self.queries = nn.Parameter(torch.zeros(1, num_bins, embed_dim))

        self.blocks = nn.ModuleList([
            CylinderDecoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        self.norm_out = nn.LayerNorm(embed_dim)

        self.occ_head = nn.Linear(embed_dim, 1)
        self.radius_head = nn.Linear(embed_dim, 1)
        self.local_norm_q = nn.LayerNorm(embed_dim)
        self.local_norm_kv = nn.LayerNorm(embed_dim)
        
        self.width_head = nn.Linear(embed_dim, 1)
        self.depth_head = nn.Linear(embed_dim + 1, 1)
        
        self.local_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        nn.init.trunc_normal_(self.queries, std=0.02)

    def _aligned_image_features(self, patches):
        B, P, D = patches.shape
        assert P == self.grid_h * self.grid_w

        # [B, 128, D] -> [B, 8, 16, D]
        patch_grid = patches.reshape(
            B,
            self.grid_h,
            self.grid_w,
            D,
        )

        dtype = patches.dtype
        device = patches.device

        fov = torch.tensor(
            self.fov_degrees * torch.pi / 180.0,
            device=device,
            dtype=dtype,
        )

        # Angle for each cylinder bin
        theta = torch.linspace(
            -0.5 * fov,
            0.5 * fov,
            self.num_bins,
            device=device,
            dtype=dtype,
        )

        # Cylinder angle -> image x
        x = (
            0.5
            + 0.5
            * torch.tan(theta)
            / torch.tan(0.5 * fov)
        )

        # Nearest image column for each cylinder bin
        col = torch.clamp(
            torch.round(x * self.grid_w - 0.5).long(),
            0,
            self.grid_w - 1,
        )

        # For every cylinder bin, collect all 8 vertical patches
        # from its corresponding image column.
        #
        # local_patches:
        # [B, num_bins, grid_h, D]
        local_patches = patch_grid[:, :, col, :].permute(
            0, 2, 1, 3
        )

        # Treat every cylinder bin as its own small attention problem:
        #
        # query:  [B*N, 1, D]
        # keys:   [B*N, 8, D]
        q = self.queries.expand(B, -1, -1).reshape(B * self.num_bins, 1, D)
        kv = local_patches.reshape(B * self.num_bins, self.grid_h, D)

        qn = self.local_norm_q(q)
        kvn = self.local_norm_kv(kv)

        aligned, _ = self.local_attn(qn, kvn, kvn, need_weights=False)
        aligned = aligned.reshape(B, self.num_bins, D)

        return aligned

    def predict_depth(self, q, width):
        depth_input = torch.cat([q, width.unsqueeze(-1),], dim=-1)

        log_depth = self.depth_head(depth_input).squeeze(-1)
        log_depth = torch.clamp(log_depth, min=-2.0, max=5.0)

        return torch.exp(log_depth)

    def forward(self, patches, return_features=False):
        B = patches.shape[0]

        # Explicit visuell feature för rätt spatial position.
        aligned = self._aligned_image_features(patches)

        # Learned bin-query + spatialt alignad bildinformation.
        q = self.queries.expand(B, -1, -1) + aligned

        # Queries får fortfarande använda hela bilden.
        for block in self.blocks:
            q = block(q, patches)

        q = self.norm_out(q)

        occupancy = torch.sigmoid(self.occ_head(q).squeeze(-1))
        radius = F.softplus(self.radius_head(q).squeeze(-1))

        # Predicted apparent angular width in degrees.
        width = F.softplus(self.width_head(q).squeeze(-1))

        depth = self.predict_depth(q, width)

        vision = torch.stack([occupancy, radius, depth, width], dim=-1)

        if return_features:
            return vision, q

        return vision

class CorrespondenceDecoder(nn.Module):
    def __init__(
        self,
        embed_dim=192,
        corr_dim=64,
        num_heads=4,
        grid_h=8,
        grid_w=16,
        output_cols=16,
    ):
        super().__init__()

        self.grid_h = grid_h
        self.grid_w = grid_w
        self.output_cols = output_cols

        self.norm_a = nn.LayerNorm(embed_dim)
        self.norm_b = nn.LayerNorm(embed_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True,
        )

        self.norm_out = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, corr_dim)

    def forward(self, patches_a, patches_b):
        a = self.norm_a(patches_a)
        b = self.norm_b(patches_b)

        # Descriptor A byggs genom att söka information i B
        delta_a, _ = self.cross_attn(a, b, b, need_weights=False)

        # Descriptor B byggs genom att söka information i A
        delta_b, _ = self.cross_attn(b, a, a, need_weights=False)

        feat_a = self.norm_out(patches_a + delta_a)
        feat_b = self.norm_out(patches_b + delta_b)

        corr_a = F.normalize(self.proj(feat_a), dim=-1)
        corr_b = F.normalize(self.proj(feat_b), dim=-1)

        B, P, D = corr_a.shape

        corr_a = corr_a.reshape(B, self.grid_h, self.grid_w, D).mean(dim=1)
        corr_b = corr_b.reshape(B, self.grid_h, self.grid_w, D).mean(dim=1)

        # [B, grid_w, D] -> [B, D, grid_w]
        corr_a = corr_a.transpose(1, 2)
        corr_b = corr_b.transpose(1, 2)

        # Behåll 16 coarse correspondence columns
        corr_a = F.adaptive_avg_pool1d(corr_a, self.output_cols)
        corr_b = F.adaptive_avg_pool1d(corr_b, self.output_cols)

        corr_a = corr_a.transpose(1, 2)
        corr_b = corr_b.transpose(1, 2)

        corr_a = F.normalize(corr_a, dim=-1)
        corr_b = F.normalize(corr_b, dim=-1)

        return corr_a, corr_b

    def _patches_to_bins(self, corr, num_bins=128, grid_h=8, grid_w=16):
        B, P, D = corr.shape

        corr = corr.reshape(B, grid_h, grid_w, D).mean(dim=1)

        # Image x order -> cylinder-angle order
        corr = torch.flip(corr, dims=[1])

        corr = corr.transpose(1, 2)

        corr = F.interpolate(
            corr,
            size=num_bins,
            mode="linear",
            align_corners=True,
        )

        corr = corr.transpose(1, 2)

        return F.normalize(corr, dim=-1)


@torch.no_grad()
def gt_bin_matches(target_a, target_b, occ_thresh=0.5):
    """IDs match GT bins; ambiguous IDs are excluded as in the pose experiments."""
    if target_a.ndim != 3 or target_b.ndim != 3 or target_a.shape[-1] < 4 or target_b.shape[-1] < 4:
        raise ValueError('Expected GT tensors [batch, bins, >=4] with IDs in channel 3')
    result = []
    for a, b in zip(target_a, target_b):
        ia = torch.where(a[:, 0] > occ_thresh)[0]
        ib = torch.where(b[:, 0] > occ_thresh)[0]
        ids_a, ids_b = a[ia, 3].round().long(), b[ib, 3].round().long()
        rows = []
        for cid in ids_a.unique():
            ma, mb = ia[ids_a == cid], ib[ids_b == cid]
            if ma.numel() == 1 and mb.numel() == 1:
                rows.append(torch.stack((ma[0], mb[0])))
        result.append(torch.stack(rows) if rows else torch.empty((0, 2), device=a.device, dtype=torch.long))
    return result



def _predicted_bin_matches_numpy(a, b, da, db, fov_degrees=90.,
                                 occ_thresh=0.2, nms_radius=2,
                                 max_candidates=32, min_similarity=0.2,
                                 radius_weight=0.25, min_margin=0.0):
    """Discrete matcher on small CPU arrays; no GT or camera pose inputs.

    Select occupancy peaks, map angular cylinder bins to image columns using
    pinhole projection, score column descriptors with a relative-radius penalty,
    and keep mutual best matches. Margins reject ambiguous associations.
    Coarse descriptors are shared within a column: this is a baseline matcher,
    not a learned fine-resolution cylinder correspondence decoder.
    """
    if not (0. < fov_degrees < 180.):
        raise ValueError('Matching requires 0 < fov_degrees < 180')
    if not (0 <= occ_thresh <= 1 and -1 <= min_similarity <= 1):
        raise ValueError('Invalid occupancy or cosine similarity threshold')
    if (not isinstance(nms_radius, int) or nms_radius < 0
            or not isinstance(max_candidates, int) or max_candidates < 2):
        raise ValueError('nms_radius must be integer >=0; max_candidates integer >=2')
    if not np.isfinite([radius_weight, min_margin]).all() or min(radius_weight, min_margin) < 0:
        raise ValueError('Radius weight and match margin must be finite and nonnegative')
    for v, d in ((a, da), (b, db)):
        if v.ndim != 2 or len(v) < 2 or v.shape[1] < 3:
            raise ValueError('Vision must have shape [num_bins>=2, channels>=3]')
        if d.ndim != 2 or len(d) < 1:
            raise ValueError('Descriptors must have shape [columns, features]')
    if da.shape[1] != db.shape[1] or da.shape[1] < 1:
        raise ValueError('Descriptor feature dimensions must match')

    def select(v):
        finite = np.isfinite(v[:, :3]).all(axis=1)
        candidates = np.flatnonzero(finite & (v[:, 0] > occ_thresh)
                                   & (v[:, 1] > 0) & (v[:, 2] > 0))
        # Stable deterministic nonmaximum suppression, including flat plateaus.
        order = candidates[np.argsort(-v[candidates, 0], kind='stable')]
        kept = []
        for i in order:
            if all(abs(int(i) - j) > nms_radius for j in kept):
                kept.append(int(i))
                if len(kept) == max_candidates:
                    break
        return np.asarray(sorted(kept), dtype=np.int64)

    ia, ib = select(a), select(b)
    empty = np.empty((0, 2), dtype=np.int64)
    if len(ia) == 0 or len(ib) == 0:
        return empty

    def descriptors(v, d, indices):
        half = math.radians(fov_degrees) / 2
        theta = -half + 2 * half * indices / (len(v) - 1)
        x = 0.5 + 0.5 * np.tan(theta) / math.tan(half)
        col = np.floor(x * len(d)).astype(np.int64).clip(0, len(d) - 1)
        desc = d[col].astype(np.float64)
        norm = np.linalg.norm(desc, axis=-1, keepdims=True)
        ok = np.isfinite(desc).all(axis=-1) & np.isfinite(norm[:, 0]) & (norm[:, 0] > 1e-8)
        return desc / np.maximum(norm, 1e-8), ok

    fa, oka = descriptors(a, da, ia)
    fb, okb = descriptors(b, db, ib)
    ia, ib, fa, fb = ia[oka], ib[okb], fa[oka], fb[okb]
    if len(ia) == 0 or len(ib) == 0:
        return empty
    similarity = np.clip(fa @ fb.T, -1, 1)
    ra, rb = a[ia, 1, None], b[None, ib, 1]
    radius_cost = np.abs(ra - rb) / np.maximum(ra + rb, 1e-8)
    score = similarity - radius_weight * radius_cost
    allowed = similarity >= min_similarity
    score = np.where(allowed, score, -np.inf)
    best_b, best_a = score.argmax(axis=1), score.argmax(axis=0)
    matches = []
    for i, j in enumerate(best_b):
        if not allowed[i, j] or best_a[j] != i:
            continue
        if min_margin > 0:
            row = np.delete(score[i], j)
            col = np.delete(score[:, j], i)
            second = max(row.max(initial=-np.inf), col.max(initial=-np.inf))
            if score[i, j] - second < min_margin:
                continue
        matches.append((ia[i], ib[j]))
    return np.asarray(matches, dtype=np.int64).reshape(-1, 2)


@torch.no_grad()
def predicted_bin_matches(vision_a, vision_b, corr_a, corr_b, **kwargs):
    """Return one [K,2] index tensor per sample, without GT supervision.

    Small detached arrays are matched on CPU. This discrete selection has no
    gradient; pack_matched_pairs gathers the original tensors afterwards.
    """
    if (vision_a.ndim != 3 or vision_b.ndim != 3
            or corr_a.ndim != 3 or corr_b.ndim != 3):
        raise ValueError('Expected batched vision and descriptor tensors')
    if not (len(vision_a) == len(vision_b) == len(corr_a) == len(corr_b)):
        raise ValueError('Matching batch sizes must agree')
    arrays = [v.detach().float().cpu().numpy()
              for v in (vision_a, vision_b, corr_a, corr_b)]
    return [torch.as_tensor(_predicted_bin_matches_numpy(a, b, da, db, **kwargs),
                            device=vision_a.device, dtype=torch.long)
            for a, b, da, db in zip(*arrays)]


def pack_matched_pairs(vision_a, vision_b, matches, fov_degrees=90.):
    """Gather prediction values while preserving their autograd connections.

    matches: list of [K,2] index tensors, or [B,K,2] with [-1,-1] padding.
    Returns padded tokens, real-pair mask, and padded bin index pairs.
    """
    if len(matches) != len(vision_a) or len(vision_a) != len(vision_b):
        raise ValueError('Batch sizes do not match')
    device = vision_a.device
    half = math.radians(fov_degrees) / 2
    theta_a = torch.linspace(-half, half, vision_a.shape[1], device=device, dtype=torch.float32)
    theta_b = torch.linspace(-half, half, vision_b.shape[1], device=device, dtype=torch.float32)
    tokens, indices, lengths = [], [], []
    for batch_i, pairs in enumerate(matches):
        pairs = torch.as_tensor(pairs, device=device, dtype=torch.long).reshape(-1, 2)
        if ((pairs[:, 0] < 0) != (pairs[:, 1] < 0)).any():
            raise ValueError('Padded matches must use [-1,-1]')
        pairs = pairs[(pairs >= 0).all(dim=1)]
        ia, ib = pairs[:, 0], pairs[:, 1]
        if (ia >= vision_a.shape[1]).any() or (ib >= vision_b.shape[1]).any():
            raise ValueError('Match index outside predicted bins')
        if ia.unique().numel() != len(ia) or ib.unique().numel() != len(ib):
            raise ValueError('Matches must be one to one')
        token = torch.stack((
            theta_a[ia], vision_a[batch_i, ia, 1].float(), vision_a[batch_i, ia, 2].float(),
            theta_b[ib], vision_b[batch_i, ib, 1].float(), vision_b[batch_i, ib, 2].float(),
        ), dim=-1)
        lengths.append(len(token))
        # A masked placeholder gives empty samples a representable padded shape.
        tokens.append(token if len(token) else vision_a.new_zeros((1, 6)).float())
        indices.append(pairs if len(pairs) else torch.full((1, 2), -1, device=device, dtype=torch.long))
    padded = pad_sequence(tokens, batch_first=True)
    padded_indices = pad_sequence(indices, batch_first=True, padding_value=-1)
    lengths = torch.tensor(lengths, device=device)
    valid = torch.arange(padded.shape[1], device=device).unsqueeze(0) < lengths.unsqueeze(1)
    return padded, valid, padded_indices


class PairImageCylinderModel(nn.Module):
    """Matched-pair pose with optional GT-free matching.

    Output ordering and state_dict keys match the previous version.
    <2 matched pairs => zero pose, pose_valid=False.
    Explicit matching_mode='gt' permits GT-assisted diagnostics in eval.
    """
    def __init__(self, img_size=128, patch_size=(16, 8), in_chans=3,
                 embed_dim=192, depth=4, num_heads=4, num_bins=128,
                 dropout=0., mlp_ratio=3., num_register_tokens=2,
                 pose_embed_dim=128, pose_num_heads=4, pose_depth=2,
                 pose_dropout=0.1, fov_degrees=90., radius_scale=1.,
                 distance_scale=10.):
        super().__init__()
        self.fov_degrees = fov_degrees
        self.backbone = PairViTBackbone(
            img_size, patch_size, in_chans, embed_dim, depth, num_heads,
            dropout, num_register_tokens, mlp_ratio,
        )
        self.vision_head = CylinderDecoder(
            embed_dim=embed_dim, num_heads=num_heads, num_bins=num_bins,
            grid_h=self.backbone.patch_embed.grid_h,
            grid_w=self.backbone.patch_embed.grid_w,
            mlp_ratio=mlp_ratio, dropout=dropout, depth=3, fov_degrees=fov_degrees,
        )
        self.corr_decoder = CorrespondenceDecoder(
            embed_dim=embed_dim, corr_dim=64, num_heads=num_heads,
            grid_h=self.backbone.patch_embed.grid_h,
            grid_w=self.backbone.patch_embed.grid_w, output_cols=16,
        )
        with torch.random.fork_rng(devices=[]):
            self.pose_head = MatchedPairPoseModel(
                embed_dim=pose_embed_dim, num_heads=pose_num_heads, depth=pose_depth,
                dropout=pose_dropout, fov_degrees=fov_degrees,
                radius_scale=radius_scale, distance_scale=distance_scale,
            )

    def forward(self, img_a, img_b, return_attention=False, return_corr=False,
                return_features=False, compute_pose=True, target_vision_a=None,
                target_vision_b=None, matches=None, return_pose_inputs=False,
                occ_thresh=0.5, matching_mode="auto",
                pred_occ_thresh=0.2, match_nms_radius=2,
                match_max_candidates=32, match_min_similarity=0.2,
                match_radius_weight=0.25, match_min_margin=0.0):
        # Auto never consumes GT targets in eval mode.
        if matching_mode not in ("auto", "gt", "predicted", "explicit"):
            raise ValueError('matching_mode must be auto, gt, predicted, or explicit')
        source = matching_mode
        if source == "auto":
            if matches is not None:
                source = "explicit"
            elif self.training and (target_vision_a is not None or target_vision_b is not None):
                source = "gt"
            else:
                source = "predicted"
        if compute_pose and matches is not None and source != "explicit":
            raise ValueError('Explicit matches conflict with requested matching_mode')
        backbone = self.backbone(img_a, img_b, return_attention=return_attention)
        _, _, patches_a, patches_b = backbone[:4]
        va, fa = self.vision_head(patches_a, return_features=True)
        vb, fb = self.vision_head(patches_b, return_features=True)
        pose, auxiliary = None, None
        corr = self.corr_decoder(patches_a, patches_b) if (
            return_corr or (compute_pose and source == "predicted")
        ) else None
        if compute_pose:
            if source == "gt":
                if target_vision_a is None or target_vision_b is None:
                    raise ValueError('GT matching requires both target_vision_a and target_vision_b')
                if target_vision_a.shape[:2] != va.shape[:2] or target_vision_b.shape[:2] != vb.shape[:2]:
                    raise ValueError('GT and predicted bin counts must agree')
                matches = gt_bin_matches(target_vision_a, target_vision_b, occ_thresh)
            elif source == "predicted":
                matches = predicted_bin_matches(
                    va, vb, *corr, fov_degrees=self.fov_degrees,
                    occ_thresh=pred_occ_thresh, nms_radius=match_nms_radius,
                    max_candidates=match_max_candidates,
                    min_similarity=match_min_similarity,
                    radius_weight=match_radius_weight, min_margin=match_min_margin,
                )
            elif matches is None:
                raise ValueError('Explicit matching requires matches')
            tokens, valid, indices = pack_matched_pairs(va, vb, matches, self.fov_degrees)
            eligible = valid.sum(dim=1) >= 2
            # No all-masked attention: mask a dummy token in otherwise empty rows.
            safe = valid.clone()
            safe[~safe.any(dim=1), 0] = True
            pose = self.pose_head(tokens, safe)
            pose = torch.where(eligible.unsqueeze(-1), pose, torch.zeros_like(pose))
            auxiliary = dict(pair_tokens=tokens, pair_mask=valid,
                             match_indices=indices, pose_valid=eligible,
                             match_source=source, num_matches=valid.sum(dim=1))
        outputs = (va, vb, pose)
        if return_corr:
            outputs += corr
        if return_features:
            outputs += (fa, fb)
        if return_attention:
            outputs += backbone[4:]
        if return_pose_inputs:
            outputs += (auxiliary,)
        return outputs

    @torch.no_grad()
    def predict_pose_from_images(self, img_a, img_b, **matching_kwargs):
        """Eval-only callback: returns (pose, pose_valid), with no GT inputs.

        Use e.g. evaluate_pose_forest(..., predict_pose=lambda m, a, b:
            m.predict_pose_from_images(a, b, pred_occ_thresh=0.2)).
        Call model.eval() before invoking this method.
        """
        if self.training:
            raise RuntimeError('Call model.eval() before predict_pose_from_images')
        forbidden = {"matching_mode", "matches", "target_vision_a", "target_vision_b",
                     "compute_pose", "return_pose_inputs", "return_corr",
                     "return_attention", "return_features"}
        if forbidden.intersection(matching_kwargs):
            raise ValueError('Pass only predicted-matching threshold/settings here')
        _, _, pose, info = self(
            img_a, img_b, matching_mode="predicted", return_pose_inputs=True,
            **matching_kwargs,
        )
        return pose, info["pose_valid"]

    def load_vision_checkpoint(self, path):
        saved = torch.load(path, map_location='cpu', weights_only=False)
        state = saved.get('model_state_dict', saved)
        state = {k: v for k, v in state.items()
                 if k.startswith(('backbone.', 'vision_head.', 'corr_decoder.'))}
        expected = {k for k in self.state_dict()
                    if k.startswith(('backbone.', 'vision_head.', 'corr_decoder.'))}
        if set(state) != expected:
            raise ValueError('Vision checkpoint does not contain exactly the expected vision modules')
        self.load_state_dict(state, strict=False)  # Shape differences still raise.

    def load_pose_checkpoint(self, path):
        saved = torch.load(path, map_location='cpu', weights_only=False)
        cfg = saved.get('model_config', {})
        if cfg and abs(cfg['fov_degrees'] - self.fov_degrees) > 1e-6:
            raise ValueError('Pose checkpoint field of view differs from the model')
        self.pose_head.load_state_dict(saved.get('model_state_dict', saved), strict=True)
