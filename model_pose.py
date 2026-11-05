"""RGB -> predicted cylinders -> matched-pair pose.

matching_mode='auto': train uses GT matches when both targets are supplied;
eval always uses predicted matches, even when GT targets are supplied.
Train without targets also uses predicted matches. Explicit matches override
auto; matching_mode='gt' is an intentional diagnostic, including in eval.

Matching uses discrete selection (no gradient). Selected radius/depth retain
gradients to the vision branch. No new trainable parameters are introduced.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from model import PairViTBackbone, CylinderDecoder, CorrespondenceDecoder
from matched_pair_pose import MatchedPairPoseModel


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
