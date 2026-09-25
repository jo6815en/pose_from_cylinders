import torch
import torch.nn as nn
import torch.nn.functional as F


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
            in_chans,
            embed_dim,
            kernel_size=(patch_h, patch_w),
            stride=(patch_h, patch_w),
        )

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=3.0, dropout=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim), nn.Dropout(dropout),)

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

        da, _ = self.self_attn(an, an, an, need_weights=False,)
        db, _ = self.self_attn(bn, bn, bn, need_weights=False,)

        a, b = a + da, b + db

        aq, bq = self.norm_cross_q(a), self.norm_cross_q(b)
        akv, bkv = self.norm_cross_kv(a), self.norm_cross_kv(b)

        da, attn_ab = self.cross_attn(aq, bkv, bkv, need_weights=return_attention, average_attn_weights=False,)

        db, attn_ba = self.cross_attn(bq, akv, akv, need_weights=return_attention, average_attn_weights=False,)

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
        self.blocks = nn.ModuleList([PairBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)])
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


class CylinderDecoder(nn.Module):
    """One learned query per cylinder bin; deliberately only one decoder layer for laptop use."""
    def __init__(self, embed_dim=192, num_heads=4, num_bins=128, mlp_ratio=3.0, dropout=0.0):
        super().__init__()
        self.queries = nn.Parameter(torch.zeros(1, num_bins, embed_dim))
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
        self.norm_out = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, 3)
        nn.init.trunc_normal_(self.queries, std=0.02)

    def forward(self, patches):
        q = self.queries.expand(patches.shape[0], -1, -1)
        qn, kv = self.norm_q(q), self.norm_kv(patches)
        dq, _ = self.cross_attn(qn, kv, kv, need_weights=False)
        q = q + dq
        q = q + self.mlp(self.norm_mlp(q))
        y = self.output(self.norm_out(q))
        return torch.stack([torch.sigmoid(y[..., 0]), F.softplus(y[..., 1]), F.softplus(y[..., 2]),], dim=-1)


class PoseHead(nn.Module):
    def __init__(self, embed_dim=192, grid_h=8, grid_w=16, hidden_dim=192,
                 temperature=0.1, depth_scale=20.0):
        super().__init__()

        self.grid_h = grid_h
        self.grid_w = grid_w
        self.temperature = temperature
        self.depth_scale = depth_scale

        # 4 correspondence features + 6 depth/matched-depth features + yaw
        trans_input_dim = 10 * grid_w + 2

        self.translation_head = nn.Sequential(
            nn.Linear(trans_input_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

        self.yaw_head = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def _column_depth(self, vision):
        B, N, _ = vision.shape
        gw = self.grid_w

        assert N % gw == 0, "num_bins must be divisible by grid_w"
        bins_per_col = N // gw

        occ = vision[..., 0].reshape(B, gw, bins_per_col)
        depth = vision[..., 2].reshape(B, gw, bins_per_col)

        weights = occ * (occ > 0.5).float()
        mass = weights.sum(dim=-1)

        depth_col = (weights * depth).sum(dim=-1) / (mass + 1e-6)
        valid_col = (mass > 1e-6).float()

        # Cylinder-angle order is opposite to image x / patch-column order
        depth_col = torch.flip(depth_col, dims=[1])
        valid_col = torch.flip(valid_col, dims=[1])

        # Keep absolute depth information, only rescale numerically
        depth_col = depth_col / self.depth_scale

        return depth_col, valid_col

    def forward(self, cam_a, cam_b, corr_a, corr_b, vision_a, vision_b):
        B, P, D = corr_a.shape
        gh, gw = self.grid_h, self.grid_w

        assert P == gh * gw, f"Expected {gh * gw} patches, got {P}"

        # 8x16 patch features -> 16 horizontal column features
        a = corr_a.reshape(B, gh, gw, D).mean(dim=1)
        b = corr_b.reshape(B, gh, gw, D).mean(dim=1)

        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)

        # A <-> B correspondence probabilities over 16 horizontal columns
        sim = torch.matmul(a, b.transpose(1, 2)) / self.temperature
        prob_ab = F.softmax(sim, dim=-1)
        prob_ba = F.softmax(sim.transpose(1, 2), dim=-1)

        coords = torch.linspace(
            -1.0, 1.0, gw,
            device=corr_a.device,
            dtype=prob_ab.dtype,
        )

        expected_b = torch.matmul(prob_ab, coords)
        expected_a = torch.matmul(prob_ba, coords)

        disp_ab = expected_b - coords
        disp_ba = expected_a - coords

        conf_ab = prob_ab.max(dim=-1).values
        conf_ba = prob_ba.max(dim=-1).values

        # Yaw from global camera tokens
        yaw_feat = torch.cat([cam_a, cam_b], dim=-1)
        yaw = F.normalize(self.yaw_head(yaw_feat), dim=-1)

        # One depth value per horizontal column
        depth_a, valid_a = self._column_depth(vision_a)
        depth_b, valid_b = self._column_depth(vision_b)

        depth_a = depth_a.to(prob_ab.dtype)
        depth_b = depth_b.to(prob_ab.dtype)
        valid_a = valid_a.to(prob_ab.dtype)
        valid_b = valid_b.to(prob_ab.dtype)

        # Couple depth explicitly to the learned A <-> B correspondence
        weights_ab = prob_ab * valid_b.unsqueeze(1)
        weights_ba = prob_ba * valid_a.unsqueeze(1)

        denom_ab = weights_ab.sum(dim=-1)
        denom_ba = weights_ba.sum(dim=-1)

        matched_depth_b = torch.matmul(
            weights_ab, depth_b.unsqueeze(-1)
        ).squeeze(-1) / (denom_ab + 1e-6)

        matched_depth_a = torch.matmul(
            weights_ba, depth_a.unsqueeze(-1)
        ).squeeze(-1) / (denom_ba + 1e-6)

        match_valid_ab = valid_a * (denom_ab > 1e-6).to(valid_a.dtype)
        match_valid_ba = valid_b * (denom_ba > 1e-6).to(valid_b.dtype)

        matched_depth_b = matched_depth_b * match_valid_ab
        matched_depth_a = matched_depth_a * match_valid_ba

        depth_diff_ab = (matched_depth_b - depth_a) * match_valid_ab
        depth_diff_ba = (matched_depth_a - depth_b) * match_valid_ba

        trans_feat = torch.cat([
            disp_ab,
            disp_ba,
            conf_ab,
            conf_ba,
            depth_a,
            matched_depth_b,
            depth_diff_ab,
            depth_b,
            matched_depth_a,
            depth_diff_ba,
            yaw,
        ], dim=-1)

        translation = F.normalize(self.translation_head(trans_feat), dim=-1)
        return torch.cat([translation, yaw], dim=-1)


class PairImageCylinderModel(nn.Module):
    """Laptop-sized counterpart of model_v2 with the same public class/output interface as the old model.py."""
    def __init__(self, img_size=128, patch_size=(16,8), in_chans=3, embed_dim=192,
                 depth=4, num_heads=4, num_bins=128, dropout=0.0,
                 mlp_ratio=3.0, num_register_tokens=2):
        super().__init__()
        self.backbone = PairViTBackbone(
            img_size, patch_size, in_chans, embed_dim, depth, num_heads,
            dropout, num_register_tokens, mlp_ratio,
        )
        self.vision_head = CylinderDecoder(embed_dim, num_heads, num_bins, mlp_ratio, dropout)
        self.corr_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 64))
        self.pose_head = PoseHead(
            embed_dim=embed_dim,
            grid_h=self.backbone.patch_embed.grid_h,
            grid_w=self.backbone.patch_embed.grid_w,
        )

    def forward(self, img_a, img_b, return_attention=False, return_corr=False, pose_vision_a=None, pose_vision_b=None):
        if return_attention:
            cam_a, cam_b, patches_a, patches_b, attn_ab_all, attn_ba_all = self.backbone(img_a, img_b, return_attention=True)
        else:
            cam_a, cam_b, patches_a, patches_b = self.backbone(img_a, img_b)

        vision_a = self.vision_head(patches_a)
        vision_b = self.vision_head(patches_b)

        corr_a = self.corr_head(patches_a)
        corr_b = self.corr_head(patches_b)

        pose_input_a = vision_a.detach() if pose_vision_a is None else pose_vision_a
        pose_input_b = vision_b.detach() if pose_vision_b is None else pose_vision_b

        pose_ab = self.pose_head(
            cam_a,
            cam_b,
            corr_a,
            corr_b,
            pose_input_a,
            pose_input_b,
        )       
        outputs = (vision_a, vision_b, pose_ab)

        if return_corr:
            outputs += (corr_a, corr_b)

        if return_attention:
            outputs += (attn_ab_all, attn_ba_all)

        return outputs

if __name__ == "__main__":
    model = PairImageCylinderModel()
    a = torch.randn(2, 3, 128, 128)
    b = torch.randn(2, 3, 128, 128)
    va, vb, pose = model(a, b)
    print("vision_a:", va.shape)
    print("vision_b:", vb.shape)
    print("pose:", pose.shape)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
