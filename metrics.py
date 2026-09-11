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
