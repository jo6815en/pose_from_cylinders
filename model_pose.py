"""Pair cylinder model with a directly trained relative-pose prediction."""

import torch
from torch import nn
from torch.nn import functional as F

from model import PairViTBackbone, CylinderDecoder, CorrespondenceDecoder


class RelativePoseHead(nn.Module):
    """Predict camera B in camera A coordinates: [tx, ty, sin(yaw), cos(yaw)].

    Translation is a unit direction (the current supervision has no metric scale).
    The backbone has already mixed the two views through cross attention.
    """

    def __init__(self, embed_dim=192, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(6 * embed_dim),
            nn.Linear(6 * embed_dim, 2 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 4),
        )
        # A valid initial rotation and translation direction.
        nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor([1.0, 0.0, 0.0, 1.0]))

    def forward(self, camera_a, camera_b, patches_a, patches_b):
        pooled_a = patches_a.mean(dim=1)
        pooled_b = patches_b.mean(dim=1)
        features = torch.cat(
            [camera_a, camera_b, pooled_a, pooled_b,
             camera_b - camera_a, pooled_b - pooled_a], dim=-1
        )
        raw = self.net(features.float())
        direction = F.normalize(raw[:, :2], dim=-1, eps=1e-6)
        yaw = F.normalize(raw[:, 2:], dim=-1, eps=1e-6)
        return torch.cat((direction, yaw), dim=-1)


class PairImageCylinderModel(nn.Module):
    """Drop-in interface: forward returns (vision_a, vision_b, pose_ab).

    Optional return_corr, return_features and return_attention append the same
    outputs in the same order as the original model. No RANSAC is invoked.
    """

    def __init__(self, img_size=128, patch_size=(16, 8), in_chans=3,
                 embed_dim=192, depth=4, num_heads=4, num_bins=128,
                 dropout=0.0, mlp_ratio=3.0, num_register_tokens=2):
        super().__init__()
        self.backbone = PairViTBackbone(
            img_size, patch_size, in_chans, embed_dim, depth, num_heads,
            dropout, num_register_tokens, mlp_ratio,
        )
        self.vision_head = CylinderDecoder(
            embed_dim=embed_dim, num_heads=num_heads, num_bins=num_bins,
            grid_h=self.backbone.patch_embed.grid_h,
            grid_w=self.backbone.patch_embed.grid_w,
            mlp_ratio=mlp_ratio, dropout=dropout, depth=3,
        )
        self.corr_decoder = CorrespondenceDecoder(
            embed_dim=embed_dim, corr_dim=64, num_heads=num_heads,
            grid_h=self.backbone.patch_embed.grid_h,
            grid_w=self.backbone.patch_embed.grid_w, output_cols=16,
        )
        #self.pose_head = RelativePoseHead(embed_dim, dropout)
        # Skapa posehuvudet utan att ändra slumpsekvensen
        # för resten av träningskörningen.
        with torch.random.fork_rng(devices=[]):
            self.pose_head = RelativePoseHead(embed_dim, dropout)
            
    def forward(self, img_a, img_b, return_attention=False,
                return_corr=False, return_features=False, compute_pose=True,):
        backbone_out = self.backbone(img_a, img_b,
                                     return_attention=return_attention)
        camera_a, camera_b, patches_a, patches_b = backbone_out[:4]
        vision_a, feat_a = self.vision_head(patches_a, return_features=True)
        vision_b, feat_b = self.vision_head(patches_b, return_features=True)
        
        pose_ab = None

        if compute_pose:
            pose_ab = self.pose_head(
                camera_a,
                camera_b,
                patches_a,
                patches_b,
            )
        #pose_ab = self.pose_head(camera_a, camera_b, patches_a, patches_b)
        
        outputs = (vision_a, vision_b, pose_ab)
        if return_corr:
            outputs += self.corr_decoder(patches_a, patches_b)
        if return_features:
            outputs += (feat_a, feat_b)
        if return_attention:
            outputs += backbone_out[4:]
        return outputs
