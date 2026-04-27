import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    def __init__(self, img_size=128, patch_size=16, in_chans=3, embed_dim=256):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class PairViTBackbone(nn.Module):
    def __init__(
        self,
        img_size=128,
        patch_size=16,
        in_chans=3,
        embed_dim=256,
        depth=4,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        seq_len = 1 + 2 * num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.view_embed = nn.Parameter(torch.zeros(1, 2, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, dropout=dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.view_embed, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, img_a, img_b):
        tok_a = self.patch_embed(img_a) + self.view_embed[:, 0:1, :]
        tok_b = self.patch_embed(img_b) + self.view_embed[:, 1:2, :]

        x = torch.cat([tok_a, tok_b], dim=1)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x[:, 0]


class CylinderHead(nn.Module):
    def __init__(self, embed_dim=256, num_cylinders=4):
        super().__init__()
        self.num_cylinders = num_cylinders
        self.out_dim = num_cylinders * 5  # (cx, cy, cz, radius, confidence)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, self.out_dim),
        )

    def forward(self, x):
        y = self.mlp(x)
        y = y.view(x.shape[0], self.num_cylinders, 5)

        center = torch.tanh(y[..., 0:3])      # [-1, 1]
        radius = F.softplus(y[..., 3:4])      # > 0
        confidence = torch.sigmoid(y[..., 4:5])  # [0, 1]

        return torch.cat([center, radius, confidence], dim=-1)


class PairImageCylinderModel(nn.Module):
    def __init__(
        self,
        img_size=128,
        patch_size=16,
        in_chans=3,
        embed_dim=256,
        depth=4,
        num_heads=4,
        num_cylinders=4,
        dropout=0.1,
    ):
        super().__init__()
        self.backbone = PairViTBackbone(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.head = CylinderHead(embed_dim=embed_dim, num_cylinders=num_cylinders)

    def forward(self, img_a, img_b):
        feat = self.backbone(img_a, img_b)
        return self.head(feat)

