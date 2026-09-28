import torch


@torch.no_grad()
def pose_errors(pred_pose, gt_pose):
    pred_t = pred_pose[:, :2]
    gt_t = gt_pose[:, :2]

    # Total XY translation error
    translation_error = torch.linalg.vector_norm(
        pred_t - gt_t,
        dim=-1,
    )

    # Error in translation magnitude
    pred_mag = torch.linalg.vector_norm(pred_t, dim=-1)
    gt_mag = torch.linalg.vector_norm(gt_t, dim=-1)

    translation_magnitude_error = (
        pred_mag - gt_mag
    ).abs()

    # Error in translation direction
    pred_angle = torch.atan2(pred_t[:, 1], pred_t[:, 0])
    gt_angle = torch.atan2(gt_t[:, 1], gt_t[:, 0])

    direction_error = torch.atan2(
        torch.sin(pred_angle - gt_angle),
        torch.cos(pred_angle - gt_angle),
    ).abs()

    translation_direction_error = torch.rad2deg(
        direction_error
    )

    # Yaw error
    pred_yaw = torch.atan2(pred_pose[:, 2], pred_pose[:, 3])
    gt_yaw = torch.atan2(gt_pose[:, 2], gt_pose[:, 3])

    yaw_error = torch.atan2(
        torch.sin(pred_yaw - gt_yaw),
        torch.cos(pred_yaw - gt_yaw),
    ).abs()

    yaw_error_deg = torch.rad2deg(yaw_error)

    return (
        translation_error.mean(),
        translation_magnitude_error.mean(),
        translation_direction_error.mean(),
        yaw_error_deg.mean(),
    )


def relative_cylinder_errors(pred_vision, target_vision, occ_thresh=0.5, fov_degrees=90.0):
    B, N, _ = target_vision.shape
    mask = target_vision[..., 0] > occ_thresh
    valid = mask.any(dim=1)

    pred_radius, gt_radius = pred_vision[..., 1], target_vision[..., 1]
    radius_num = (((pred_radius - gt_radius) ** 2) * mask).sum(dim=1).sqrt()
    radius_den = ((gt_radius ** 2) * mask).sum(dim=1).sqrt().clamp_min(1e-6)
    radius_err = radius_num / radius_den

    theta = torch.linspace(-0.5 * torch.pi * fov_degrees / 180.0, 0.5 * torch.pi * fov_degrees / 180.0, N, device=target_vision.device, dtype=target_vision.dtype)
    direction = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)

    pred_pos = pred_vision[..., 2].unsqueeze(-1) * direction
    gt_pos = target_vision[..., 2].unsqueeze(-1) * direction
    pos_num = ((((pred_pos - gt_pos) ** 2).sum(dim=-1)) * mask).sum(dim=1).sqrt()
    pos_den = (((gt_pos ** 2).sum(dim=-1) * mask).sum(dim=1)).sqrt().clamp_min(1e-6)
    position_err = pos_num / pos_den

    if valid.any():
        return radius_err[valid].mean(), position_err[valid].mean()
    zero = pred_vision.sum() * 0.0
    return zero, zero


def relative_pose_vector_errors(pred_pose, target_pose):
    pred_t, gt_t = pred_pose[:, :2], target_pose[:, :2]
    pred_r, gt_r = pred_pose[:, 2:], target_pose[:, 2:]

    translation_err = torch.linalg.vector_norm(pred_t - gt_t, dim=-1) / torch.linalg.vector_norm(gt_t, dim=-1).clamp_min(1e-6)
    rotation_err = torch.linalg.vector_norm(pred_r - gt_r, dim=-1) / torch.linalg.vector_norm(gt_r, dim=-1).clamp_min(1e-6)

    return translation_err.mean(), rotation_err.mean()