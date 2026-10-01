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
        self.depth_head = nn.Linear(embed_dim, 1)
        self.local_norm_q = nn.LayerNorm(embed_dim)
        self.local_norm_kv = nn.LayerNorm(embed_dim)

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
            - 0.5
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

        aligned, _ = self.local_attn(
            qn,
            kvn,
            kvn,
            need_weights=False,
        )

        aligned = aligned.reshape(B, self.num_bins, D)

        return aligned

    def forward(self, patches):
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

        log_depth = self.depth_head(q).squeeze(-1)

        # Begränsa till ett numeriskt rimligt område
        log_depth = torch.clamp(log_depth, min=-2.0, max=5.0)

        # Resten av modellen får fortfarande vanlig metrisk depth
        depth = torch.exp(log_depth)

        return torch.stack([occupancy, radius, depth], dim=-1)

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
        delta_a, _ = self.cross_attn(a, b, b, need_weights=False)

        # Descriptor B byggs genom att söka information i A
        delta_b, _ = self.cross_attn(b, a, a, need_weights=False)

        feat_a = self.norm_out(patches_a + delta_a)
        feat_b = self.norm_out(patches_b + delta_b)

        corr_a = F.normalize(self.proj(feat_a), dim=-1)
        corr_b = F.normalize(self.proj(feat_b), dim=-1)

        B, P, D = corr_a.shape

        # 8x16 patches -> 16 horizontal descriptors
        corr_a = corr_a.reshape(B, 8, 16, D).mean(dim=1)
        corr_b = corr_b.reshape(B, 8, 16, D).mean(dim=1)

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
        self.grid_w = grid_w
        self.temperature = temperature
        self.fov_degrees = fov_degrees

    def _sample_geometry(self, vision, theta):
        """
        Sample depth + occupancy continuously from the 128 cylinder bins.
        theta: [B, M] in radians
        """
        B, N, _ = vision.shape
        fov = self.fov_degrees * torch.pi / 180.0

        # theta -> continuous cylinder-bin coordinate
        u = (theta + 0.5 * fov) / fov * (N - 1)
        u = u.clamp(0, N - 1)

        i0 = torch.floor(u).long()
        i1 = torch.clamp(i0 + 1, max=N - 1)
        alpha = u - i0.to(u.dtype)

        depth = vision[..., 2]
        occ = vision[..., 0]

        d0 = torch.gather(depth, 1, i0)
        d1 = torch.gather(depth, 1, i1)
        o0 = torch.gather(occ, 1, i0)
        o1 = torch.gather(occ, 1, i1)

        sampled_depth = (1.0 - alpha) * d0 + alpha * d1
        sampled_occ = (1.0 - alpha) * o0 + alpha * o1

        points = torch.stack([
            sampled_depth * torch.cos(theta), sampled_depth * torch.sin(theta),
        ], dim=-1)

        return points, sampled_occ

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
        B = corr_a.shape[0]
        device = corr_a.device
        dtype = corr_a.dtype

        # Do not let pose loss dominate correspondence learning
        corr_a = F.normalize(corr_a.detach(), dim=-1)
        corr_b = F.normalize(corr_b.detach(), dim=-1)

        # 16x16 visual correspondence
        sim = torch.matmul(corr_a, corr_b.transpose(1, 2)) / self.temperature
        prob_ab = F.softmax(sim, dim=-1)

        if gt_prob_ab is not None:
            prob_ab = gt_prob_ab.to(device=device, dtype=dtype)

        # Image-column centres
        x = (torch.arange(self.grid_w, device=device, dtype=dtype) + 0.5) / self.grid_w

        fov = torch.tensor(
            self.fov_degrees * torch.pi / 180.0,
            device=device,
            dtype=dtype,
        )

        # Image x -> cylinder angle
        theta_a = torch.atan((1.0 - 2.0 * x) * torch.tan(0.5 * fov))

        theta_a = theta_a.unsqueeze(0).expand(B, -1)

        # Local soft expectation around the strongest match
        peak = prob_ab.argmax(dim=-1)

        coords = torch.arange(self.grid_w, device=device, dtype=dtype)

        distance = torch.abs(coords.view(1, 1, self.grid_w) - peak.unsqueeze(-1))

        mask = (distance <= 1).to(prob_ab.dtype)

        local_prob = prob_ab * mask
        local_prob = local_prob / (local_prob.sum(dim=-1, keepdim=True) + 1e-8)

        # Continuous image coordinate [0, 1]
        x_coords = (torch.arange(self.grid_w, device=device, dtype=dtype) + 0.5) / self.grid_w

        x_b = torch.matmul(local_prob, x_coords)

        # Continuous expected B angle
        theta_b = torch.atan((1.0 - 2.0 * x_b) * torch.tan(0.5 * fov))

        # Sample full 128-bin geometry at these continuous angles
        points_a, occ_a = self._sample_geometry(vision_a, theta_a)
        points_b, occ_b = self._sample_geometry(vision_b, theta_b)

        confidence = prob_ab.max(dim=-1).values
        weights = occ_a * occ_b * confidence
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)

        # Weighted centroids
        centroid_a = (points_a * weights.unsqueeze(-1)).sum(dim=1)
        centroid_b = (points_b * weights.unsqueeze(-1)).sum(dim=1)

        ac = points_a - centroid_a.unsqueeze(1)
        bc = points_b - centroid_b.unsqueeze(1)

        ax, ay = ac[..., 0], ac[..., 1]
        bx, by = bc[..., 0], bc[..., 1]

        dot = (weights * (ax * bx + ay * by)).sum(dim=1)
        cross = (weights * (ax * by - ay * bx)).sum(dim=1)

        norm = torch.sqrt(dot.square() + cross.square() + 1e-8)

        cos_yaw = dot / norm
        sin_yaw = cross / norm

        yaw = torch.stack([sin_yaw, cos_yaw], dim=-1)

        ca_x = centroid_a[:, 0]
        ca_y = centroid_a[:, 1]

        rotated_a = torch.stack([
            cos_yaw * ca_x - sin_yaw * ca_y, sin_yaw * ca_x + cos_yaw * ca_y,
        ], dim=-1)

        translation = centroid_b - rotated_a
        translation = F.normalize(translation, dim=-1)

        return torch.cat([translation, yaw], dim=-1)

class RansacPoseEstimator(nn.Module):
    def __init__(
        self,
        search_radius=16,
        temperature=0.1,
        num_iters=200,
        inlier_threshold=0.4,
        occ_thresh=0.5,
        fov_degrees=90.0,
    ):
        super().__init__()
        self.search_radius = search_radius
        self.temperature = temperature
        self.num_iters = num_iters
        self.inlier_threshold = inlier_threshold
        self.occ_thresh = occ_thresh
        self.fov_degrees = fov_degrees

    @staticmethod
    def _rigid_transform(A, B):
        ca = A.mean(dim=0)
        cb = B.mean(dim=0)

        Ac = A - ca
        Bc = B - cb

        dot = (Ac[:, 0] * Bc[:, 0] + Ac[:, 1] * Bc[:, 1]).sum()
        cross = (Ac[:, 0] * Bc[:, 1] - Ac[:, 1] * Bc[:, 0]).sum()

        norm = torch.sqrt(dot.square() + cross.square() + 1e-8)
        c = dot / norm
        s = cross / norm

        R = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])

        t = cb - R @ ca
        return R, t, s, c

    def _ransac(self, A, B):
        n = A.shape[0]

        if n < 2:
            return None

        best_inliers = None
        best_count = 0
        best_error = float("inf")

        for _ in range(self.num_iters):
            idx = torch.randperm(n, device=A.device)[:2]
            R, t, _, _ = self._rigid_transform(A[idx], B[idx])

            pred_B = A @ R.T + t
            residuals = torch.linalg.vector_norm(pred_B - B, dim=-1)

            inliers = residuals < self.inlier_threshold
            count = inliers.sum().item()

            if count < 2:
                continue

            error = residuals[inliers].mean().item()

            if count > best_count or (count == best_count and error < best_error):
                best_count = count
                best_error = error
                best_inliers = inliers

        if best_inliers is None or best_inliers.sum() < 2:
            return None

        R, t, s, c = self._rigid_transform(A[best_inliers], B[best_inliers])
        return t, s, c

    def forward(self, vision_a, vision_b, corr_a, corr_b):
        corr_a = F.normalize(corr_a, dim=-1)
        corr_b = F.normalize(corr_b, dim=-1)

        prob = F.softmax(
            torch.matmul(corr_a, corr_b.transpose(1, 2)) / self.temperature, dim=-1
        )

        B, N, _ = vision_a.shape
        device = vision_a.device
        dtype = vision_a.dtype

        fov = torch.tensor(
            self.fov_degrees * torch.pi / 180.0,
            device=device,
            dtype=dtype,
        )

        theta = torch.linspace(-0.5 * fov, 0.5 * fov, N, device=device, dtype=dtype)

        pts_a = torch.stack([
            vision_a[..., 2] * torch.cos(theta),
            vision_a[..., 2] * torch.sin(theta),
        ], dim=-1)

        pts_b = torch.stack([
            vision_b[..., 2] * torch.cos(theta),
            vision_b[..., 2] * torch.sin(theta),
        ], dim=-1)

        poses = []

        for b in range(B):
            # Use soft predicted occupancy at inference.
            occ_a = vision_a[b, :, 0]
            occ_b = vision_b[b, :, 0]

            # Select strongest predicted cylinders in A/B rather than
            # requiring occupancy > 0.5.
            active_a = torch.where(occ_a > 0.05)[0]
            active_b_mask = occ_b > 0.05

            matched_a = []
            matched_b = []

            for ia in active_a:
                theta_a = theta[ia]

                x_a = 0.5 - 0.5 * torch.tan(theta_a) / torch.tan(0.5 * fov)
                col_a = torch.clamp((x_a * 16).long(), 0, 15)

                # Coarse visual match
                col_b = prob[b, col_a].argmax()

                # Coarse B column -> cylinder-bin neighbourhood
                x_b = (col_b.to(dtype) + 0.5) / 16.0
                theta_b = torch.atan(
                    (1.0 - 2.0 * x_b) * torch.tan(0.5 * fov)
                )

                center = torch.round(
                    (theta_b + 0.5 * fov) / fov * (N - 1)
                ).long().clamp(0, N - 1)

                lo = max(0, center.item() - self.search_radius)
                hi = min(N, center.item() + self.search_radius + 1)

                candidates = torch.arange(lo, hi, device=device)
                candidates = candidates[active_b_mask[candidates]]

                if candidates.numel() == 0:
                    continue

                # Fine matching by predicted cylinder radius
                radius_a = vision_a[b, ia, 1]
                radius_diff = torch.abs(vision_b[b, candidates, 1] - radius_a)
                ib = candidates[radius_diff.argmin()]

                matched_a.append(pts_a[b, ia])
                matched_b.append(pts_b[b, ib])

            if len(matched_a) < 2:
                poses.append(torch.zeros(4, device=device, dtype=dtype))
                continue

            A = torch.stack(matched_a)
            Bpts = torch.stack(matched_b)

            result = self._ransac(A, Bpts)

            if result is None:
                poses.append(torch.zeros(4, device=device, dtype=dtype))
                continue

            t, s, c = result
            t = F.normalize(t, dim=0)

            poses.append(torch.stack([t[0], t[1], s, c]))

        return torch.stack(poses)

class PairImageCylinderModel(nn.Module):
    """Laptop-sized model with the same public interface as the old model."""
    def __init__(
        self, img_size=128, patch_size=(16, 8), in_chans=3, embed_dim=192,
        depth=4, num_heads=4, num_bins=128, dropout=0.0,
        mlp_ratio=3.0, num_register_tokens=2,
    ):
        super().__init__()
        self.backbone = PairViTBackbone(
            img_size, patch_size, in_chans, embed_dim, depth, num_heads,
            dropout, num_register_tokens, mlp_ratio,
        )
        self.vision_head = CylinderDecoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_bins=num_bins,
            grid_h=self.backbone.patch_embed.grid_h,
            grid_w=self.backbone.patch_embed.grid_w,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            depth=3,
        )
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
        self.ransac_pose = RansacPoseEstimator(
            search_radius=16,
            temperature=0.1,
            num_iters=200,
            inlier_threshold=0.4,
        )

    def forward(
        self, img_a, img_b, return_attention=False, return_corr=False,
        pose_vision_a=None, pose_vision_b=None, gt_prob_ab=None, gt_prob_ba=None,
    ):
        if return_attention:
            cam_a, cam_b, patches_a, patches_b, attn_ab_all, attn_ba_all = self.backbone(
                img_a, img_b, return_attention=True
            )
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

    @torch.no_grad()
    def predict_pose_ransac(self, img_a, img_b):
        cam_a, cam_b, patches_a, patches_b = self.backbone(img_a, img_b)

        vision_a = self.vision_head(patches_a)
        vision_b = self.vision_head(patches_b)

        corr_a, corr_b = self.corr_decoder(patches_a, patches_b)

        pose = self.ransac_pose(
            vision_a,
            vision_b,
            corr_a,
            corr_b,
        )

        return vision_a, vision_b, pose

if __name__ == "__main__":
    model = PairImageCylinderModel()
    a = torch.randn(2, 3, 128, 128)
    b = torch.randn(2, 3, 128, 128)
    va, vb, pose = model(a, b)
    print("vision_a:", va.shape)
    print("vision_b:", vb.shape)
    print("pose:", pose.shape)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
