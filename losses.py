import torch
import torch.nn.functional as F
import math

def vision_to_cylinder_centers(
    vision: torch.Tensor,
    fov_degrees: float = 90.0,
    depth_channel: int = 2,
    occ_channel: int = 0,
) -> torch.Tensor:
    """
    Converts vision bins to 2D center points.

    Assumes:
        vision shape: [B, N, C] or [N, C]
        occ_channel contains one-hot/soft occupancy over bins
        depth_channel contains depth d per bin

    Returns:
        centers: [B, 2] or [2]
    """
    squeeze_batch = False
    if vision.dim() == 2:
        vision = vision.unsqueeze(0)
        squeeze_batch = True

    num_bins = vision.shape[1]
    device = vision.device
    dtype = vision.dtype

    occ = vision[..., occ_channel]
    depth = vision[..., depth_channel]

    bin_idx = torch.argmax(occ, dim=1)

    fov = math.radians(fov_degrees)
    theta_min = -0.5 * fov
    theta_max = 0.5 * fov

    theta_bins = torch.linspace(
        theta_min,
        theta_max,
        num_bins,
        device=device,
        dtype=dtype,
    )

    theta = theta_bins[bin_idx]
    d = depth[torch.arange(vision.shape[0], device=device), bin_idx]

    centers = torch.stack(
        [
            d * torch.cos(theta),
            d * torch.sin(theta),
        ],
        dim=-1,
    )

    if squeeze_batch:
        centers = centers.squeeze(0)

    return centers


def match_vision_to_target_cylinders(
    vision: torch.Tensor,
    target_cylinders: torch.Tensor,
    fov_degrees: float = 90.0,
    depth_channel: int = 2,
    occ_channel: int = 0,
) -> dict:
    """
    Matches predicted/target vision center to closest GT cylinder.

    target_cylinders shape:
        [M, 4] or [B, M, 4]
        columns: x, y, r, h
    """
    pred_centers = vision_to_cylinder_centers(
        vision,
        fov_degrees=fov_degrees,
        depth_channel=depth_channel,
        occ_channel=occ_channel,
    )

    squeeze_batch = False
    if target_cylinders.dim() == 2:
        target_cylinders = target_cylinders.unsqueeze(0)
        pred_centers = pred_centers.unsqueeze(0)
        squeeze_batch = True

    gt_centers = target_cylinders[..., :2]

    distances = torch.cdist(
        pred_centers.unsqueeze(1),
        gt_centers,
    ).squeeze(1)

    matched_idx = torch.argmin(distances, dim=1)
    matched_distance = distances[
        torch.arange(distances.shape[0], device=distances.device),
        matched_idx,
    ]

    matched_cylinders = target_cylinders[
        torch.arange(target_cylinders.shape[0], device=target_cylinders.device),
        matched_idx,
    ]

    result = {
        "pred_centers": pred_centers,
        "matched_indices": matched_idx,
        "matched_distances": matched_distance,
        "matched_cylinders": matched_cylinders,
    }

    if squeeze_batch:
        result = {
            key: value.squeeze(0) if value.dim() > 0 else value
            for key, value in result.items()
        }

    return result

def add_sup_losses(
        total_vis,
        total_radius,
        total_pose,
        total_depth,
        total_t,
        total_r,
        loss_out
):
    total_vis += loss_out["vision_a"].item()
    total_radius += loss_out["vision_a_radius"].item()
    total_pose += loss_out["pose"].item()
    total_depth += loss_out["vision_a_depth"].item()
    total_t += loss_out["translation"].item()
    total_r += loss_out["rotation"].item()

    return total_vis, total_radius, total_pose, total_depth, total_t, total_r,

def add_con_losses(
        total_con,
        total_occ,
        total_rad,
        total_dep,
        loss_out
):
    total_con += loss_out["total"].item()
    total_occ += loss_out["occ"].item()
    total_rad += loss_out["radius"].item()
    total_dep += loss_out["depth"].item()
    
    return total_con, total_occ, total_rad, total_dep

def add_reproj_losses(
        total_reproj,
        total_occ,
        total_rad,
        total_dep,
        loss_out
):
    total_reproj += loss_out["total"].item()
    total_occ += loss_out["occ"].item()
    total_rad += loss_out["radius"].item()
    total_dep += loss_out["depth"].item()
    
    return total_reproj, total_occ, total_rad, total_dep


def vision_loss(
    pred_vision,
    target_vision,
    occ_thresh=0.5,
    lambda_occ=1.0,
    lambda_radius=1.0,
    lambda_depth=1.0,
):
    """
    pred_vision, target_vision: (B, num_bins, 3)
        channel 0 = occupancy
        channel 1 = radius
        channel 2 = depth
    """

    pred_occ = pred_vision[..., 0]
    pred_rad = pred_vision[..., 1]
    pred_dep = pred_vision[..., 2]

    tgt_occ = target_vision[..., 0]
    tgt_rad = target_vision[..., 1]
    tgt_dep = target_vision[..., 2]

    # 1) Occupancy: BCE
    occ_loss = F.binary_cross_entropy(
        pred_occ.clamp(1e-6, 1.0 - 1e-6),
        tgt_occ
    )

    # 2) Radius + depth only where occupancy is on
    mask = (tgt_occ > occ_thresh).float()

    rad_l1 = F.l1_loss(pred_rad, tgt_rad, reduction="none")
    dep_l1 = F.l1_loss(pred_dep, tgt_dep, reduction="none")

    rad_loss = (rad_l1 * mask).sum() / mask.sum().clamp_min(1.0)
    dep_loss = (dep_l1 * mask).sum() / mask.sum().clamp_min(1.0)

    total = (
        lambda_occ * occ_loss +
        lambda_radius * rad_loss +
        lambda_depth * dep_loss
    )

    return total, occ_loss, rad_loss, dep_loss


def relative_pose_loss_2d(
    pred_pose,
    target_pose,
    t_weight=1.0,
    r_weight=1.0,
):
    """
    pred_pose, target_pose: (B, 4) = [tx, ty, sin(yaw), cos(yaw)]
    """

    pred_t = pred_pose[:, :2]
    tgt_t = target_pose[:, :2]

    pred_r = F.normalize(pred_pose[:, 2:], dim=-1)
    tgt_r = F.normalize(target_pose[:, 2:], dim=-1)

    t_loss = F.smooth_l1_loss(pred_t, tgt_t)
    r_loss = F.mse_loss(pred_r, tgt_r)

    total = t_weight * t_loss + r_weight * r_loss
    return total, t_loss, r_loss


def supervised_loss(
    pred_vision_a,
    target_vision_a,
    pred_vision_b,
    target_vision_b,
    pred_pose,
    target_pose,
    lambda_pose=1.0,
    lambda_vis=1.0,
    occ_thresh=0.5,
    lambda_occ=1.0,
    lambda_radius=1.0,
    lambda_depth=1.0,
    t_weight=1.0,
    r_weight=1.0,
):    
    pred_vis_a = pred_vision_a
    pred_vis_b = pred_vision_b

    vis_a_total, vis_a_occ, vis_a_rad, vis_a_dep = vision_loss(
        pred_vis_a,
        target_vision_a,
        occ_thresh=occ_thresh,
        lambda_occ=lambda_occ,
        lambda_radius=lambda_radius,
        lambda_depth=lambda_depth,
    )

    vis_b_total, vis_b_occ, vis_b_rad, vis_b_dep = vision_loss(
        pred_vis_b,
        target_vision_b,
        occ_thresh=occ_thresh,
        lambda_occ=lambda_occ,
        lambda_radius=lambda_radius,
        lambda_depth=lambda_depth,
    )


    pose_total, t_loss, r_loss = relative_pose_loss_2d(
        pred_pose,
        target_pose,
        t_weight=t_weight,
        r_weight=r_weight,
    )

    total = (lambda_vis * vis_a_total + lambda_vis * vis_b_total)/2.0  + lambda_pose * pose_total

    return {
        "total": total,
        "vision_a": vis_a_total,
        "vision_a_occ": vis_a_occ,
        "vision_a_radius": vis_a_rad,
        "vision_a_depth": vis_a_dep,
        "pose": pose_total,
        "translation": t_loss,
        "rotation": r_loss,
    }


def consistency_loss(
    pred_vision_1,
    pred_vision_2,
    occ_thresh=0.5,
    match_thresh=0.25,
    lambda_occ=1.0,
    lambda_radius=1.0,
    lambda_depth=1.0,
):
    """
    pred_vision_1, pred_vision_2: (B, num_bins, 3)
        channel 0 = occupancy
        channel 1 = radius
        channel 2 = depth

    Jämför bara cylindrar som verkar motsvara samma objekt.
    Extra cylindrar i ena vyn ignoreras.
    """

    total_loss = pred_vision_1.new_tensor(0.0)
    total_occ = pred_vision_1.new_tensor(0.0)
    total_rad = pred_vision_1.new_tensor(0.0)
    total_dep = pred_vision_1.new_tensor(0.0)

    batch_size = pred_vision_1.shape[0]
    valid_batches = 0

    for b in range(batch_size):
        v1 = pred_vision_1[b]
        v2 = pred_vision_2[b]

        occ1 = v1[:, 0]
        rad1 = v1[:, 1]
        dep1 = v1[:, 2]

        occ2 = v2[:, 0]
        rad2 = v2[:, 1]
        dep2 = v2[:, 2]

        m1 = occ1 > occ_thresh
        m2 = occ2 > occ_thresh

        if m1.sum() == 0 or m2.sum() == 0:
            continue

        f1 = torch.stack([rad1[m1], dep1[m1]], dim=-1)  # (N1, 2)
        f2 = torch.stack([rad2[m2], dep2[m2]], dim=-1)  # (N2, 2)

        o1 = occ1[m1]
        o2 = occ2[m2]

        # Pairwise cost mellan möjliga cylinder-matchningar
        cost = torch.cdist(f1, f2, p=1)  # (N1, N2)

        # Mutual nearest neighbors
        nn12 = cost.argmin(dim=1)  # for each in v1 -> best in v2
        nn21 = cost.argmin(dim=0)  # for each in v2 -> best in v1

        matched_i = []
        matched_j = []

        for i in range(cost.shape[0]):
            j = nn12[i].item()
            if nn21[j].item() == i and cost[i, j] < match_thresh:
                matched_i.append(i)
                matched_j.append(j)

        if len(matched_i) == 0:
            continue

        idx1 = torch.tensor(matched_i, device=pred_vision_1.device)
        idx2 = torch.tensor(matched_j, device=pred_vision_1.device)

        # Occupancy consistency för de matchade cylindrarna
        loss_occ = F.mse_loss(o1[idx1], o2[idx2])

        # Radius + depth consistency
        loss_rad = F.l1_loss(f1[idx1, 0], f2[idx2, 0])
        loss_dep = F.l1_loss(f1[idx1, 1], f2[idx2, 1])

        loss = (
            lambda_occ * loss_occ +
            lambda_radius * loss_rad +
            lambda_depth * loss_dep
        )

        total_loss = total_loss + loss
        total_occ = total_occ + loss_occ
        total_rad = total_rad + loss_rad
        total_dep = total_dep + loss_dep
        valid_batches += 1

    if valid_batches == 0:
        zero = pred_vision_1.new_tensor(0.0)
        return {
            "total": zero,
            "occ": zero,
            "radius": zero,
            "depth": zero,
        }

    total_loss = total_loss / valid_batches
    total_occ = total_occ / valid_batches
    total_rad = total_rad / valid_batches
    total_dep = total_dep / valid_batches

    return {
        "total": total_loss,
        "occ": total_occ,
        "radius": total_rad,
        "depth": total_dep,
    }


def reprojection_loss_2d(
    vision_a,
    vision_b,
    pose_ab,
    occ_thresh=0.5,
    lambda_occ=1.0,
    lambda_radius=1.0,
    lambda_depth=1.0,
    u_min=-1.0,
    u_max=1.0,
):
    """
    vision_a, vision_b: (B, N, 3) = [occ, radius, depth]
    pose_ab: (B, 4) = [tx, ty, sin(yaw), cos(yaw)]

    Matchar generatorns vision-binning:
    u = x / y
    där x = lateral/sida, y = framåt/depth.

    u_min=-1, u_max=1 motsvarar ungefär 90 graders FOV.
    """
    B, N, _ = vision_a.shape
    device = vision_a.device
    dtype = vision_a.dtype

    occ_a = vision_a[..., 0]
    rad_a = vision_a[..., 1]
    dep_a = vision_a[..., 2]

    occ_b = vision_b[..., 0]
    rad_b = vision_b[..., 1]
    dep_b = vision_b[..., 2]

    # Bin centers i generatorns u-koordinat
    bin_size = (u_max - u_min) / N
    u_centers = torch.linspace(
        u_min + 0.5 * bin_size,
        u_max - 0.5 * bin_size,
        N,
        device=device,
        dtype=dtype,
    )

    u_a = u_centers.view(1, N).expand(B, N)

    # Avprojicera från u + depth till BEV
    # u = x / y, depth = y
    y_a = dep_a
    x_a = u_a * dep_a

    tx = pose_ab[:, 0].view(B, 1)
    ty = pose_ab[:, 1].view(B, 1)
    s = pose_ab[:, 2].view(B, 1)
    c = pose_ab[:, 3].view(B, 1)

    # A -> B
    x_b = c * x_a - s * y_a + tx
    y_b = s * x_a + c * y_a + ty

    # Projicera till B:s u-koordinat
    dep_proj = y_b
    u_proj = x_b / (y_b + 1e-6)

    # u -> bin-index
    bin_pos = (u_proj - u_min) / (u_max - u_min) * N - 0.5

    j0 = torch.floor(bin_pos).long()
    j1 = j0 + 1

    w1 = bin_pos - j0.float()
    w0 = 1.0 - w1

    valid = (
        (occ_a > occ_thresh)
        & (dep_a > 0)
        & (dep_proj > 0)
        & (u_proj >= u_min)
        & (u_proj <= u_max)
        & (j0 >= 0)
        & (j1 < N)
    )

    j0 = j0.clamp(0, N - 1)
    j1 = j1.clamp(0, N - 1)

    occ0 = occ_b.gather(1, j0)
    occ1 = occ_b.gather(1, j1)

    rad0 = rad_b.gather(1, j0)
    rad1 = rad_b.gather(1, j1)

    dep0 = dep_b.gather(1, j0)
    dep1 = dep_b.gather(1, j1)

    occ_sample = w0 * occ0 + w1 * occ1
    rad_sample = w0 * rad0 + w1 * rad1
    dep_sample = w0 * dep0 + w1 * dep1

    valid_f = valid.float()
    denom = valid_f.sum().clamp_min(1.0)

    occ_loss = ((1.0 - occ_sample) ** 2 * valid_f).sum() / denom

    radius_loss = (
        torch.abs(rad_sample - rad_a) * valid_f
    ).sum() / denom

    depth_loss = (
        torch.abs(dep_sample - dep_proj) / (dep_a.abs() + 1e-6) * valid_f
    ).sum() / denom

    total = (
        lambda_occ * occ_loss
        + lambda_radius * radius_loss
        + lambda_depth * depth_loss
    )

    return {
        "total": total,
        "occ": occ_loss,
        "radius": radius_loss,
        "depth": depth_loss,
    }