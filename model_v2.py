import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# Patch embedding
# ---------------------------------------------------------

class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size=256,
        patch_size=16,
        in_chans=3,
        embed_dim=512,
    ):
        super().__init__()

        assert img_size % patch_size == 0

        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

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


# ---------------------------------------------------------
# MLP
# ---------------------------------------------------------

class MLP(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()

        hidden_dim = int(dim * mlp_ratio)

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------
# Pair transformer block
#
# 1. self attention independently in A and B
# 2. cross attention A <- B and B <- A
# 3. MLP independently
# ---------------------------------------------------------

class PairBlock(nn.Module):
    def __init__(
        self,
        dim=512,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()

        # Shared self-attention weights for both views.
        self.norm_self = nn.LayerNorm(dim)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Shared cross-attention module.
        self.norm_cross_q = nn.LayerNorm(dim)
        self.norm_cross_kv = nn.LayerNorm(dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Shared MLP.
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = MLP(
            dim=dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    def forward(self, a, b):

        # -------------------------------------------------
        # Self attention
        # -------------------------------------------------

        a_norm = self.norm_self(a)
        b_norm = self.norm_self(b)

        a_self, _ = self.self_attn(
            a_norm,
            a_norm,
            a_norm,
            need_weights=False,
        )

        b_self, _ = self.self_attn(
            b_norm,
            b_norm,
            b_norm,
            need_weights=False,
        )

        a = a + a_self
        b = b + b_self

        # -------------------------------------------------
        # Cross attention
        #
        # Important: compute both from the same pre-cross
        # representations so update order does not matter.
        # -------------------------------------------------

        a_q = self.norm_cross_q(a)
        b_q = self.norm_cross_q(b)

        a_kv = self.norm_cross_kv(a)
        b_kv = self.norm_cross_kv(b)

        a_cross, _ = self.cross_attn(
            query=a_q,
            key=b_kv,
            value=b_kv,
            need_weights=False,
        )

        b_cross, _ = self.cross_attn(
            query=b_q,
            key=a_kv,
            value=a_kv,
            need_weights=False,
        )

        a = a + a_cross
        b = b + b_cross

        # -------------------------------------------------
        # Feed-forward
        # -------------------------------------------------

        a = a + self.mlp(self.norm_mlp(a))
        b = b + self.mlp(self.norm_mlp(b))

        return a, b


# ---------------------------------------------------------
# Backbone
# ---------------------------------------------------------

class PairGeometryBackbone(nn.Module):
    def __init__(
        self,
        img_size=256,
        patch_size=16,
        in_chans=3,
        embed_dim=512,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        num_register_tokens=4,
        dropout=0.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_register_tokens = num_register_tokens

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        num_patches = self.patch_embed.num_patches

        # One camera token per image.
        self.camera_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        # Register tokens.
        self.register_tokens = nn.Parameter(
            torch.zeros(
                1,
                num_register_tokens,
                embed_dim,
            )
        )

        # Learned 2D patch position embeddings.
        #
        # RoPE can replace this later. Starting simple makes
        # debugging considerably easier.
        self.patch_pos_embed = nn.Parameter(
            torch.zeros(
                1,
                num_patches,
                embed_dim,
            )
        )

        # Explicit view identity.
        self.view_embed = nn.Parameter(
            torch.zeros(1, 2, embed_dim)
        )

        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            PairBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):

        nn.init.trunc_normal_(
            self.camera_token,
            std=0.02,
        )

        nn.init.trunc_normal_(
            self.register_tokens,
            std=0.02,
        )

        nn.init.trunc_normal_(
            self.patch_pos_embed,
            std=0.02,
        )

        nn.init.trunc_normal_(
            self.view_embed,
            std=0.02,
        )

    def make_view_tokens(self, img, view_idx):

        B = img.shape[0]

        # Image patches
        patches = self.patch_embed(img)

        patches = (
            patches
            + self.patch_pos_embed
            + self.view_embed[:, view_idx:view_idx + 1]
        )

        # Camera token
        cam = self.camera_token.expand(
            B, -1, -1
        )

        cam = (
            cam
            + self.view_embed[:, view_idx:view_idx + 1]
        )

        # Register tokens
        regs = self.register_tokens.expand(
            B, -1, -1
        )

        regs = (
            regs
            + self.view_embed[:, view_idx:view_idx + 1]
        )

        # Layout:
        #
        # [CAM, REG1, REG2, ..., PATCH1, PATCH2, ...]
        tokens = torch.cat(
            [cam, regs, patches],
            dim=1,
        )

        return self.pos_drop(tokens)

    def forward(self, img_a, img_b):

        a = self.make_view_tokens(
            img_a,
            view_idx=0,
        )

        b = self.make_view_tokens(
            img_b,
            view_idx=1,
        )

        for block in self.blocks:
            a, b = block(a, b)

        a = self.norm(a)
        b = self.norm(b)

        # Camera token
        cam_a = a[:, 0]
        cam_b = b[:, 0]

        # Remove camera + register tokens.
        patch_start = 1 + self.num_register_tokens

        patches_a = a[:, patch_start:]
        patches_b = b[:, patch_start:]

        return {
            "cam_a": cam_a,
            "cam_b": cam_b,
            "patches_a": patches_a,
            "patches_b": patches_b,
        }


# ---------------------------------------------------------
# Cylinder query decoder
# ---------------------------------------------------------

class CylinderDecoder(nn.Module):
    """
    One learned query per horizontal cylinder bin.

    Each query attends to the patch representation of
    one image and predicts:

        occupancy
        radius
        depth
    """

    def __init__(
        self,
        embed_dim=512,
        num_heads=8,
        num_bins=128,
        depth=2,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()

        self.num_bins = num_bins

        self.queries = nn.Parameter(
            torch.zeros(
                1,
                num_bins,
                embed_dim,
            )
        )

        self.layers = nn.ModuleList([
            CylinderDecoderLayer(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self.output = nn.Linear(
            embed_dim,
            3,
        )

        nn.init.trunc_normal_(
            self.queries,
            std=0.02,
        )

    def forward(self, patch_tokens):

        B = patch_tokens.shape[0]

        q = self.queries.expand(
            B, -1, -1
        )

        for layer in self.layers:
            q = layer(q, patch_tokens)

        q = self.norm(q)

        y = self.output(q)

        occ = torch.sigmoid(
            y[..., 0]
        )

        radius = F.softplus(
            y[..., 1]
        )

        depth = F.softplus(
            y[..., 2]
        )

        return torch.stack(
            [occ, radius, depth],
            dim=-1,
        )


class CylinderDecoderLayer(nn.Module):
    def __init__(
        self,
        dim=512,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()

        # Let neighboring/other bin queries communicate.
        self.norm_self = nn.LayerNorm(dim)

        self.self_attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Queries read image patches.
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.cross_attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_mlp = nn.LayerNorm(dim)

        self.mlp = MLP(
            dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    def forward(self, q, image_tokens):

        # Query self-attention
        h = self.norm_self(q)

        out, _ = self.self_attn(
            h,
            h,
            h,
            need_weights=False,
        )

        q = q + out

        # Query -> image cross-attention
        q_norm = self.norm_q(q)
        kv = self.norm_kv(image_tokens)

        out, _ = self.cross_attn(
            query=q_norm,
            key=kv,
            value=kv,
            need_weights=False,
        )

        q = q + out

        # MLP
        q = q + self.mlp(
            self.norm_mlp(q)
        )

        return q


# ---------------------------------------------------------
# Relative pose head
# ---------------------------------------------------------

class PoseHead(nn.Module):
    def __init__(
        self,
        embed_dim=512,
        hidden_dim=512,
    ):
        super().__init__()

        # We use BOTH camera tokens.
        self.mlp = nn.Sequential(
            nn.Linear(
                embed_dim * 2,
                hidden_dim,
            ),
            nn.GELU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.GELU(),

            nn.Linear(
                hidden_dim,
                4,
            ),
        )

    def forward(self, cam_a, cam_b):

        x = torch.cat(
            [cam_a, cam_b],
            dim=-1,
        )

        y = self.mlp(x)

        t = y[:, :2]

        yaw_vec = F.normalize(
            y[:, 2:],
            dim=-1,
        )

        return torch.cat(
            [t, yaw_vec],
            dim=-1,
        )


# ---------------------------------------------------------
# Full model
# ---------------------------------------------------------

class PairImageCylinderModelV2(nn.Module):
    def __init__(
        self,
        img_size=256,
        patch_size=16,
        in_chans=3,
        embed_dim=512,
        depth=12,
        num_heads=8,
        num_bins=128,
        mlp_ratio=4.0,
        num_register_tokens=4,
        cylinder_decoder_depth=2,
        dropout=0.0,
    ):
        super().__init__()

        self.backbone = PairGeometryBackbone(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_register_tokens=num_register_tokens,
            dropout=dropout,
        )

        # Same cylinder decoder is used for both views.
        self.cylinder_decoder = CylinderDecoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_bins=num_bins,
            depth=cylinder_decoder_depth,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        self.pose_head = PoseHead(
            embed_dim=embed_dim,
            hidden_dim=embed_dim,
        )

    def forward(self, img_a, img_b):

        features = self.backbone(
            img_a,
            img_b,
        )

        vision_a = self.cylinder_decoder(
            features["patches_a"]
        )

        vision_b = self.cylinder_decoder(
            features["patches_b"]
        )

        pose_ab = self.pose_head(
            features["cam_a"],
            features["cam_b"],
        )

        return (
            vision_a,
            vision_b,
            pose_ab,
        )


# ---------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------

if __name__ == "__main__":

    model = PairImageCylinderModelV2(
        img_size=256,
        patch_size=16,
        embed_dim=512,
        depth=12,
        num_heads=8,
        num_bins=128,
    )

    a = torch.randn(
        2, 3, 256, 256
    )

    b = torch.randn(
        2, 3, 256, 256
    )

    vision_a, vision_b, pose = model(a, b)

    print("vision_a:", vision_a.shape)
    print("vision_b:", vision_b.shape)
    print("pose:", pose.shape)

    n_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Parameters: {n_params / 1e6:.1f} M"
    )