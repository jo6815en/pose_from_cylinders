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
        qn = self.norm_q(q)
        kv = self.norm_kv(patches)

        # Information hämtad från bilden
        dq, _ = self.cross_attn(qn, kv, kv, need_weights=False)

        # Geometry får fortfarande använda query + image information
        q = q + dq
        q = q + self.mlp(self.norm_mlp(q))

        y = self.output(self.norm_out(q))

        return torch.stack([
            torch.sigmoid(y[..., 0]),
            F.softplus(y[..., 1]),
            F.softplus(y[..., 2]),
        ], dim=-1)


class CorrespondenceDecoder(nn.Module):
    def __init__(self, embed_dim=192, corr_dim=64, num_heads=4):
        super().__init__()

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
        delta_a, _ = self.cross_attn(
            a, b, b,
            need_weights=False,
        )

        # Descriptor B byggs genom att söka information i A
        delta_b, _ = self.cross_attn(
            b, a, a,
            need_weights=False,
        )

        feat_a = self.norm_out(patches_a + delta_a)
        feat_b = self.norm_out(patches_b + delta_b)

        corr_a = F.normalize(self.proj(feat_a), dim=-1)
        corr_b = F.normalize(self.proj(feat_b), dim=-1)

        corr_a = self._patches_to_bins(corr_a)
        corr_b = self._patches_to_bins(corr_b)

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


class PoseHead(nn.Module):
    def __init__(
        self,
        embed_dim=192,
        grid_h=8,
        grid_w=16,
        hidden_dim=192,
        temperature=0.1,
        depth_scale=20.0,
        fov_degrees=90.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.fov_degrees = fov_degrees

    def _vision_points(self, vision):
        B, N, _ = vision.shape
        fov = torch.tensor(
            self.fov_degrees * torch.pi / 180.0,
            device=vision.device,
            dtype=vision.dtype,
        )
        theta = torch.linspace(
            -0.5 * fov, 0.5 * fov, N,
            device=vision.device,
            dtype=vision.dtype,
        )

        depth = vision[..., 2]
        return torch.stack([
            depth * torch.cos(theta),
            depth * torch.sin(theta),
        ], dim=-1)

    def forward(
        self,
        cam_a,
        cam_b,
        corr_a,
        corr_b,
        vision_a,
        vision_b,
        gt_prob_ab=None,
        gt_prob_ba=None,
    ):
        # corr_a/corr_b: [B, 128, 64]
        corr_a = F.normalize(corr_a.detach(), dim=-1)
        corr_b = F.normalize(corr_b.detach(), dim=-1)

        # 128 x 128 soft correspondence matrix
        sim = torch.matmul(corr_a, corr_b.transpose(1, 2)) / self.temperature
        prob_ab = F.softmax(sim, dim=-1)

        if gt_prob_ab is not None:
            prob_ab = gt_prob_ab.to(device=prob_ab.device, dtype=prob_ab.dtype)

        # Full-resolution cylinder geometry
        points_a = self._vision_points(vision_a).to(prob_ab.dtype)
        points_b = self._vision_points(vision_b).to(prob_ab.dtype)

        valid_a = vision_a[..., 0].to(prob_ab.dtype)
        valid_b = vision_b[..., 0].to(prob_ab.dtype)

        # Remove matches to empty B bins and renormalize
        match_weights = prob_ab * valid_b.unsqueeze(1)
        match_mass = match_weights.sum(dim=-1)
        match_weights = match_weights / (match_mass.unsqueeze(-1) + 1e-6)

        # Expected corresponding 2D point in B for every A bin
        matched_b = torch.matmul(match_weights, points_b)

        # Weight reliable/valid matches
        confidence = prob_ab.max(dim=-1).values
        weights = valid_a * match_mass * confidence
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)

        # Weighted centroids
        centroid_a = (points_a * weights.unsqueeze(-1)).sum(dim=1)
        centroid_b = (matched_b * weights.unsqueeze(-1)).sum(dim=1)

        a_centered = points_a - centroid_a.unsqueeze(1)
        b_centered = matched_b - centroid_b.unsqueeze(1)

        # Optimal 2D rotation
        ax, ay = a_centered[..., 0], a_centered[..., 1]
        bx, by = b_centered[..., 0], b_centered[..., 1]

        dot = (weights * (ax * bx + ay * by)).sum(dim=1)
        cross = (weights * (ax * by - ay * bx)).sum(dim=1)

        norm = torch.sqrt(dot.square() + cross.square() + 1e-8)
        cos_yaw = dot / norm
        sin_yaw = cross / norm

        yaw = torch.stack([sin_yaw, cos_yaw], dim=-1)

        # Translation from same rigid transform
        ca_x, ca_y = centroid_a[:, 0], centroid_a[:, 1]
        rotated_centroid_a = torch.stack([
            cos_yaw * ca_x - sin_yaw * ca_y,
            sin_yaw * ca_x + cos_yaw * ca_y,
        ], dim=-1)

        translation = centroid_b - rotated_centroid_a
        translation = F.normalize(translation, dim=-1)

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
        self.pose_head = PoseHead(
            embed_dim=embed_dim,
            grid_h=self.backbone.patch_embed.grid_h,
            grid_w=self.backbone.patch_embed.grid_w,
        )
        self.corr_decoder = CorrespondenceDecoder(
            embed_dim=embed_dim,
            corr_dim=64,
            num_heads=num_heads,
        )

    def forward(self, img_a, img_b, return_attention=False, return_corr=False, pose_vision_a=None, pose_vision_b=None, gt_prob_ab=None, gt_prob_ba=None):
        if return_attention:
            cam_a, cam_b, patches_a, patches_b, attn_ab_all, attn_ba_all = self.backbone(img_a, img_b, return_attention=True)
        else:
            cam_a, cam_b, patches_a, patches_b = self.backbone(img_a, img_b)

        vision_a = self.vision_head(patches_a)
        vision_b = self.vision_head(patches_b)

        corr_a, corr_b = self.corr_decoder(
            patches_a,
            patches_b,
        )
        pose_input_a = vision_a.detach() if pose_vision_a is None else pose_vision_a
        pose_input_b = vision_b.detach() if pose_vision_b is None else pose_vision_b

        pose_ab = self.pose_head(
            cam_a,
            cam_b,
            corr_a,
            corr_b,
            pose_input_a,
            pose_input_b,
            gt_prob_ab=gt_prob_ab,
            gt_prob_ba=gt_prob_ba,
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
