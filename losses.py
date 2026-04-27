import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

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