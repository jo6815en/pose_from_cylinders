"""RGB -> cylinder geometry -> matched-pair pose, differentiable end to end.

This version still requires GT bin correspondences (or explicit bin pairs).
It does not infer correspondences from images. GT provides indices only;
radius and depth in pose inputs always come from the vision predictions.
"""
import math
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
    """Same vision modules; replace the old image-token pose head with paired pose.

    forward(img_a, img_b, target_vision_a=gt_a, target_vision_b=gt_b)
    returns (vision_a, vision_b, pose). Without GT, explicit matches are required.
    compute_pose=False permits ordinary vision-only inference.

    Original optional output ordering is retained; return_pose_inputs appends a
    dict AFTER all requested correspondence/features/attention outputs.
    Samples with <2 matches return zero pose and pose_valid=False; exclude them
    from the pose loss using masked_relative_pose_loss_2d.
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
                occ_thresh=0.5):
        backbone = self.backbone(img_a, img_b, return_attention=return_attention)
        _, _, patches_a, patches_b = backbone[:4]
        va, fa = self.vision_head(patches_a, return_features=True)
        vb, fb = self.vision_head(patches_b, return_features=True)
        pose, auxiliary = None, None
        if compute_pose:
            if matches is None:
                if target_vision_a is None or target_vision_b is None:
                    raise ValueError('Pose requires GT target_vision_a/b or explicit bin matches; use compute_pose=False for vision only')
                if target_vision_a.shape[:2] != va.shape[:2] or target_vision_b.shape[:2] != vb.shape[:2]:
                    raise ValueError('GT and predicted bin counts must agree')
                matches = gt_bin_matches(target_vision_a, target_vision_b, occ_thresh)
            tokens, valid, indices = pack_matched_pairs(va, vb, matches, self.fov_degrees)
            eligible = valid.sum(dim=1) >= 2
            # No all-masked attention: mask a dummy token in otherwise empty rows.
            safe = valid.clone()
            safe[~safe.any(dim=1), 0] = True
            pose = self.pose_head(tokens, safe)
            pose = torch.where(eligible.unsqueeze(-1), pose, torch.zeros_like(pose))
            auxiliary = dict(pair_tokens=tokens, pair_mask=valid,
                             match_indices=indices, pose_valid=eligible)
        outputs = (va, vb, pose)
        if return_corr:
            outputs += self.corr_decoder(patches_a, patches_b)
        if return_features:
            outputs += (fa, fb)
        if return_attention:
            outputs += backbone[4:]
        if return_pose_inputs:
            outputs += (auxiliary,)
        return outputs

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
