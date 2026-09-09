import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    def __init__(self, img_size=128, patch_size=16, in_chans=3, embed_dim=192):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

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

    def forward(self, a, b):
        an, bn = self.norm_self(a), self.norm_self(b)
        da, _ = self.self_attn(an, an, an, need_weights=False)
        db, _ = self.self_attn(bn, bn, bn, need_weights=False)
        a, b = a + da, b + db

        aq, bq = self.norm_cross_q(a), self.norm_cross_q(b)
        akv, bkv = self.norm_cross_kv(a), self.norm_cross_kv(b)
        da, _ = self.cross_attn(aq, bkv, bkv, need_weights=False)
        db, _ = self.cross_attn(bq, akv, akv, need_weights=False)
        a, b = a + da, b + db

        a = a + self.mlp(self.norm_mlp(a))
        b = b + self.mlp(self.norm_mlp(b))
        return a, b


class PairViTBackbone(nn.Module):
    def __init__(self, img_size=128, patch_size=16, in_chans=3, embed_dim=192,
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

    def forward(self, img_a, img_b):
        a, b = self._tokens(img_a, 0), self._tokens(img_b, 1)
        for block in self.blocks:
            a, b = block(a, b)
        a, b = self.norm(a), self.norm(b)
        start = 1 + self.num_register_tokens
        return a[:, 0], b[:, 0], a[:, start:], b[:, start:]


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
        return torch.stack([
            torch.sigmoid(y[..., 0]),
            F.softplus(y[..., 1]),
            F.softplus(y[..., 2]),
        ], dim=-1)


class PoseHead(nn.Module):
    def __init__(self, embed_dim=192):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, 4),
        )

    def forward(self, cam_a, cam_b):
        y = self.mlp(torch.cat([cam_a, cam_b], dim=-1))
        return torch.cat([y[:, :2], F.normalize(y[:, 2:], dim=-1)], dim=-1)


class PairImageCylinderModel(nn.Module):
    """Laptop-sized counterpart of model_v2 with the same public class/output interface as the old model.py."""
    def __init__(self, img_size=128, patch_size=16, in_chans=3, embed_dim=192,
                 depth=4, num_heads=4, num_bins=128, dropout=0.0,
                 mlp_ratio=3.0, num_register_tokens=2):
        super().__init__()
        self.backbone = PairViTBackbone(
            img_size, patch_size, in_chans, embed_dim, depth, num_heads,
            dropout, num_register_tokens, mlp_ratio,
        )
        self.vision_head = CylinderDecoder(embed_dim, num_heads, num_bins, mlp_ratio, dropout)
        self.pose_head = PoseHead(embed_dim)

    def forward(self, img_a, img_b):
        cam_a, cam_b, patches_a, patches_b = self.backbone(img_a, img_b)
        vision_a = self.vision_head(patches_a)
        vision_b = self.vision_head(patches_b)
        pose_ab = self.pose_head(cam_a, cam_b)
        return vision_a, vision_b, pose_ab


if __name__ == "__main__":
    model = PairImageCylinderModel()
    a = torch.randn(2, 3, 128, 128)
    b = torch.randn(2, 3, 128, 128)
    va, vb, pose = model(a, b)
    print("vision_a:", va.shape)
    print("vision_b:", vb.shape)
    print("pose:", pose.shape)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
