import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn.functional as F


def single_vision_loss(
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
    pred_vision,
    target_vision_a,
    target_vision_b,
    pred_pose,
    target_pose,
    lambda_pose=1.0,
    lambda_vis_a=1.0,
    lambda_vis_b=1.0,
    occ_thresh=0.5,
    lambda_occ=1.0,
    lambda_radius=1.0,
    lambda_depth=1.0,
    t_weight=1.0,
    r_weight=1.0,
):
    """
    Passar in både om modellen ger:
      - en enda vision-prediction: pred_vision = Tensor
      - eller två vision-predictions: pred_vision = (pred_vis_a, pred_vis_b)

    Om pred_vision är en Tensor används samma prediction mot båda targets.
    """

    if isinstance(pred_vision, (tuple, list)):
        pred_vis_a, pred_vis_b = pred_vision
        print("Yes this is correct!")
    else:
        pred_vis_a = pred_vision
        pred_vis_b = pred_vision
        print("No, this is not correct!")

    vis_a_total, vis_a_occ, vis_a_rad, vis_a_dep = single_vision_loss(
        pred_vis_a,
        target_vision_a,
        occ_thresh=occ_thresh,
        lambda_occ=lambda_occ,
        lambda_radius=lambda_radius,
        lambda_depth=lambda_depth,
    )

    vis_b_total, vis_b_occ, vis_b_rad, vis_b_dep = single_vision_loss(
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

    total = lambda_vis_a * vis_a_total + lambda_vis_b * vis_b_total + lambda_pose * pose_total

    return {
        "total": total,
        "vision_a": vis_a_total,
        "vision_b": vis_b_total,
        "vision_a_occ": vis_a_occ,
        "vision_a_radius": vis_a_rad,
        "vision_a_depth": vis_a_dep,
        "vision_b_occ": vis_b_occ,
        "vision_b_radius": vis_b_rad,
        "vision_b_depth": vis_b_dep,
        "pose": pose_total,
        "translation": t_loss,
        "rotation": r_loss,
    }


# --------------------------------
# ----- Not used now -------------
# --------------------------------

def consistency_loss(pred1, pred2, lambda_radius=1.0, lambda_conf=0.1, lambda_conf_reg=0.01):
    """
    pred1, pred2: [B, K, 5]
      cylinder = (cx, cy, cz, radius, confidence)

    Returnerar:
      scalar loss
    """
    B, K, D = pred1.shape
    assert D == 5, "Förväntar [B, K, 5]"

    total_loss = 0.0

    for b in range(B):
        c1 = pred1[b]  # [K, 5]
        c2 = pred2[b]  # [K, 5]

        center1 = c1[:, :3]          # [K, 3]
        radius1 = c1[:, 3:4]         # [K, 1]
        conf1 = c1[:, 4:5]           # [K, 1]

        center2 = c2[:, :3]
        radius2 = c2[:, 3:4]
        conf2 = c2[:, 4:5]

        # Pairwise cost matrix: [K, K]
        center_cost = ((center1[:, None, :] - center2[None, :, :]) ** 2).sum(dim=-1)
        radius_cost = ((radius1[:, None, :] - radius2[None, :, :]) ** 2).squeeze(-1)

        cost = center_cost + lambda_radius * radius_cost

        # Hungarian matching
        row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())

        # Matchade loss-termer
        matched_center = F.mse_loss(center1[row_ind], center2[col_ind])
        matched_radius = F.mse_loss(radius1[row_ind], radius2[col_ind])

        matched_conf = F.mse_loss(conf1[row_ind], conf2[col_ind])
        conf_reg = (conf1.mean() + conf2.mean()) * 0.5


        sample_loss = (
                    matched_center
                    + lambda_radius * matched_radius
                    + lambda_conf * matched_conf
                    + lambda_conf_reg * conf_reg
                )
        
        total_loss = total_loss + sample_loss

    return total_loss / B


def sinkhorn(log_alpha, n_iters=20):
    """
    log_alpha: [B, K, K]
    Returnerar en ungefär doubly-stochastic matris P med shape [B, K, K]
    """
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
    return torch.exp(log_alpha)


def consistency_loss_fast(
    pred1,
    pred2,
    lambda_radius=5.0,
    lambda_conf=0.1,
    lambda_conf_reg=0.01,
    lambda_geom=10.0,
    temperature=0.02,
    sinkhorn_iters=30,
):
    """
    pred1, pred2: [B, K, 5]
      cylinder = (cx, cy, cz, radius, confidence)

    Snabb, vectoriserad approximation till set consistency loss.
    """
    assert pred1.shape == pred2.shape
    B, K, D = pred1.shape
    assert D == 5, "Förväntar [B, K, 5]"

    c1 = pred1
    c2 = pred2

    center1 = c1[..., :3]      # [B, K, 3]
    radius1 = c1[..., 3:4]     # [B, K, 1]
    conf1 = c1[..., 4:5]       # [B, K, 1]

    center2 = c2[..., :3]
    radius2 = c2[..., 3:4]
    conf2 = c2[..., 4:5]

    # Pairwise cost per batch: [B, K, K]
    center_cost = torch.cdist(center1, center2, p=2).pow(2)
    radius_cost = (radius1.unsqueeze(2) - radius2.unsqueeze(1)).pow(2).squeeze(-1)

    cost = center_cost + lambda_radius * radius_cost

    # Soft matching matrix, batchad och GPU-vänlig
    P = sinkhorn(-cost / temperature, n_iters=sinkhorn_iters)  # [B, K, K]

    # Matcha pred2 mot pred1
    matched2 = torch.bmm(P, c2)                      # [B, K, 5]
    matched1 = torch.bmm(P.transpose(1, 2), c1)      # [B, K, 5]

    # Geometrisk consistency
    geom_loss = F.mse_loss(c1[..., :4], matched2[..., :4])

    geom_loss = lambda_geom * geom_loss

    # Confidence consistency
    conf_loss = F.mse_loss(conf1, matched2[..., 4:5])

    # Enkel regularization: håll confidence lite sparsam
    conf_reg = 0.5 * (conf1.mean() + conf2.mean())

    return geom_loss + lambda_conf * conf_loss + lambda_conf_reg * conf_reg


def diversity_loss(pred):
    centers = pred[..., :3]
    conf = pred[..., 4:5]

    B, K, _ = centers.shape

    d = torch.cdist(centers, centers, p=2)

    eye = torch.eye(K, device=pred.device).bool()
    d = d.masked_fill(eye.unsqueeze(0), float("inf"))

    weight = conf @ conf.transpose(1, 2)

    penalty = (1.0 / (d + 1e-3)) * weight

    return penalty.mean()