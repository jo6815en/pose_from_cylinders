import torch
import torch.nn.functional as F
import numpy as np
import inspect

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

import numpy as np
import torch


@torch.no_grad()
def evaluate_vision(
    model,
    loader,
    device,
    occ_threshold=0.5,
):
    model.eval()

    # Kontrollera originalmodellen även om torch.compile används.
    base_model = getattr(model, "_orig_mod", model)

    supports_compute_pose = (
        "compute_pose" in inspect.signature(base_model.forward).parameters
    )

    occ_gt_all = []
    occ_pred_all = []

    radius_gt_all = []
    radius_pred_all = []

    depth_gt_all = []
    depth_pred_all = []

    for batch in loader:
        img_a, vision_a, img_b, vision_b, _ = [
            x.to(device, non_blocking=True)
            for x in batch
        ]

        if supports_compute_pose:
            pred_a, pred_b, _ = model(img_a, img_b, compute_pose=False)
        else:
            pred_a, pred_b, _ = model(img_a, img_b)

        # Evaluate both views
        for pred, gt in [
            (pred_a, vision_a),
            (pred_b, vision_b),
        ]:
            gt_occ = gt[..., 0]
            pred_occ = pred[..., 0]

            occ_gt_all.append(gt_occ.cpu())
            occ_pred_all.append(pred_occ.cpu())

            # Geometry only where GT says a cylinder exists
            mask = gt_occ > occ_threshold

            radius_gt_all.append(gt[..., 1][mask].cpu())
            radius_pred_all.append(pred[..., 1][mask].cpu())

            depth_gt_all.append(gt[..., 2][mask].cpu())
            depth_pred_all.append(pred[..., 2][mask].cpu())

    occ_gt = torch.cat(occ_gt_all).numpy()
    occ_pred = torch.cat(occ_pred_all).numpy()

    radius_gt = torch.cat(radius_gt_all).numpy()
    radius_pred = torch.cat(radius_pred_all).numpy()

    depth_gt = torch.cat(depth_gt_all).numpy()
    depth_pred = torch.cat(depth_pred_all).numpy()

    # --------------------------------------------------
    # Occupancy
    # --------------------------------------------------

    occ_gt_binary = occ_gt > occ_threshold
    occ_pred_binary = occ_pred > occ_threshold

    occ_accuracy = np.mean(occ_pred_binary == occ_gt_binary)

    tp = np.sum(occ_pred_binary & occ_gt_binary)
    fp = np.sum(occ_pred_binary & ~occ_gt_binary)
    fn = np.sum(~occ_pred_binary & occ_gt_binary)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    f1 = (2 * precision * recall / max(precision + recall, 1e-12))

    # --------------------------------------------------
    # Radius
    # --------------------------------------------------

    radius_error = radius_pred - radius_gt

    radius_mae = np.mean(np.abs(radius_error))
    radius_rmse = np.sqrt(np.mean(radius_error ** 2))
    radius_bias = np.mean(radius_error)

    radius_corr = np.corrcoef(radius_pred, radius_gt,)[0, 1]

    # --------------------------------------------------
    # Depth
    # --------------------------------------------------

    depth_error = depth_pred - depth_gt

    depth_mae = np.mean(np.abs(depth_error))
    depth_rmse = np.sqrt(np.mean(depth_error ** 2))
    depth_bias = np.mean(depth_error)

    depth_corr = np.corrcoef(
        depth_pred,
        depth_gt,
    )[0, 1]

    # --------------------------------------------------
    # Occupancy tolerance in bins
    # --------------------------------------------------

    def occupancy_recall_with_tolerance(gt, pred, threshold, tol):
        gt_bin = gt > threshold
        pred_bin = pred > threshold

        hits = 0
        total = gt_bin.sum()

        for offset in range(-tol, tol + 1):
            if offset < 0:
                hits_mask = gt_bin[:, -offset:] & pred_bin[:, :offset]
            elif offset > 0:
                hits_mask = gt_bin[:, :-offset] & pred_bin[:, offset:]
            else:
                hits_mask = gt_bin & pred_bin

            # mark GT cylinders hit by any allowed offset
            if offset == -tol:
                matched = np.zeros_like(gt_bin, dtype=bool)

            if offset < 0:
                matched[:, -offset:] |= hits_mask
            elif offset > 0:
                matched[:, :-offset] |= hits_mask
            else:
                matched |= hits_mask

        return matched.sum() / max(total, 1)


    occ_recall_1bin = occupancy_recall_with_tolerance(
        occ_gt, occ_pred, occ_threshold, 1
    )

    occ_recall_2bin = occupancy_recall_with_tolerance(
        occ_gt, occ_pred, occ_threshold, 2
    )

    # Radius tolerance
    radius_err = np.abs(radius_pred - radius_gt)

    radius_5cm  = np.mean(radius_err <= 0.05)
    radius_10cm = np.mean(radius_err <= 0.10)
    radius_20cm = np.mean(radius_err <= 0.20)

    # Depth tolerance
    depth_err = np.abs(depth_pred - depth_gt)

    depth_05m = np.mean(depth_err <= 0.5)
    depth_1m  = np.mean(depth_err <= 1.0)
    depth_2m  = np.mean(depth_err <= 2.0)
    depth_5m  = np.mean(depth_err <= 5.0)

    # --------------------------------------------------
    # Print
    # --------------------------------------------------
    print("\nVISION PERFORMANCE")
    print("=" * 78)

    print("\nOCCUPANCY, occ_threshold=%.2f" % occ_threshold)
    print("-" * 78)
    print(
        f"{'Accuracy':>10}"
        f"{'Precision':>12}"
        f"{'Recall':>10}"
        f"{'F1':>10}"
        f"{'Recall ±1':>12}"
        f"{'Recall ±2':>12}"
    )
    print(
        f"{occ_accuracy:10.3f}"
        f"{precision:12.3f}"
        f"{recall:10.3f}"
        f"{f1:10.3f}"
        f"{occ_recall_1bin:12.3f}"
        f"{occ_recall_2bin:12.3f}"
    )


    print("\nGEOMETRY")
    print("-" * 78)
    print(
        f"{'':<10}"
        f"{'N':>8}"
        f"{'MAE':>10}"
        f"{'RMSE':>10}"
        f"{'Bias':>10}"
        f"{'Corr':>10}"
        f"{'Pred std':>10}"
        f"{'GT std':>10}"
    )

    print(
        f"{'Radius':<10}"
        f"{len(radius_gt):8d}"
        f"{radius_mae:10.3f}"
        f"{radius_rmse:10.3f}"
        f"{radius_bias:10.3f}"
        f"{radius_corr:10.3f}"
        f"{radius_pred.std():10.3f}"
        f"{radius_gt.std():10.3f}"
    )

    print(
        f"{'Depth':<10}"
        f"{len(depth_gt):8d}"
        f"{depth_mae:10.3f}"
        f"{depth_rmse:10.3f}"
        f"{depth_bias:10.3f}"
        f"{depth_corr:10.3f}"
        f"{depth_pred.std():10.3f}"
        f"{depth_gt.std():10.3f}"
    )


    print("\nTOLERANCE ACCURACY")
    print("-" * 78)

    print(
        f"{'Radius':<10}"
        f"{'±0.05 m':>12}"
        f"{'±0.10 m':>12}"
        f"{'±0.20 m':>12}"
    )

    print(
        f"{'':<10}"
        f"{radius_5cm:12.3f}"
        f"{radius_10cm:12.3f}"
        f"{radius_20cm:12.3f}"
    )

    print()

    print(
        f"{'Depth':<10}"
        f"{'±0.5 m':>12}"
        f"{'±1.0 m':>12}"
        f"{'±2.0 m':>12}"
        f"{'±5.0 m':>12}"
    )

    print(
        f"{'':<10}"
        f"{depth_05m:12.3f}"
        f"{depth_1m:12.3f}"
        f"{depth_2m:12.3f}"
        f"{depth_5m:12.3f}"
    )

    return {
        "occupancy_accuracy": occ_accuracy,
        "occupancy_precision": precision,
        "occupancy_recall": recall,
        "occupancy_f1": f1,

        "radius_mae": radius_mae,
        "radius_rmse": radius_rmse,
        "radius_bias": radius_bias,
        "radius_corr": radius_corr,

        "depth_mae": depth_mae,
        "depth_rmse": depth_rmse,
        "depth_bias": depth_bias,
        "depth_corr": depth_corr,
    }


@torch.no_grad()
def evaluate_pose(model, loader, device, gt_occ_threshold=0.5):
    model.eval()

    base_model = getattr(model, "_orig_mod", model)
    parameters = inspect.signature(base_model.forward).parameters

    uses_gt_matches = all(
        name in parameters
        for name in (
            "target_vision_a",
            "target_vision_b",
            "return_pose_inputs",
        )
    )

    if uses_gt_matches:
        method = "Learned pose with GT matches"
    elif callable(getattr(base_model, "predict_pose_ransac", None)):
        method = "RANSAC"
    else:
        method = "Direct learned pose"

    translation_errors = []
    yaw_errors = []

    failures = 0
    total = 0
    excluded_gt_pairs = 0

    for batch in loader:
        img_a, vision_a, img_b, vision_b, pose_gt = [
            x.to(device, non_blocking=True)
            for x in batch
        ]

        # GT translation direction and yaw must be defined.
        eligible = (
            torch.isfinite(pose_gt).all(dim=-1)
            & (pose_gt[:, :2].norm(dim=-1) > 1e-6)
            & (pose_gt[:, 2:].norm(dim=-1) > 1e-8)
        )

        if uses_gt_matches:
            _, _, pose_pred, pose_info = model(
                img_a,
                img_b,
                target_vision_a=vision_a,
                target_vision_b=vision_b,
                return_pose_inputs=True,
                occ_thresh=gt_occ_threshold,
            )
            eligible &= pose_info["pose_valid"]

        elif method == "RANSAC":
            _, _, pose_pred = base_model.predict_pose_ransac(
                img_a,
                img_b,
            )

        else:
            _, _, pose_pred = model(img_a, img_b)

        if pose_pred is None:
            raise ValueError("Modellen returnerade ingen pose.")

        excluded_gt_pairs += (~eligible).sum().item()

        for index in torch.where(eligible)[0].tolist():
            pred = pose_pred[index].float()
            gt = pose_gt[index].float()
            total += 1

            if (
                not torch.isfinite(pred).all()
                or pred[:2].norm() < 1e-8
                or pred[2:].norm() < 1e-8
            ):
                failures += 1
                continue

            # Translation direction error
            t_pred = F.normalize(pred[:2], dim=0)
            t_gt = F.normalize(gt[:2], dim=0)

            cos_t = torch.dot(t_pred, t_gt).clamp(-1.0, 1.0)
            t_error = torch.rad2deg(torch.acos(cos_t)).item()

            # Wrapped yaw error
            yaw_pred = torch.atan2(pred[2], pred[3])
            yaw_gt = torch.atan2(gt[2], gt[3])

            delta_yaw = torch.atan2(
                torch.sin(yaw_pred - yaw_gt),
                torch.cos(yaw_pred - yaw_gt),
            )
            yaw_error = torch.rad2deg(delta_yaw).abs().item()

            translation_errors.append(t_error)
            yaw_errors.append(yaw_error)

    t = np.asarray(translation_errors)
    yaw = np.asarray(yaw_errors)

    def stat(values, function):
        return (
            float(function(values))
            if len(values)
            else float("nan")
        )

    print("CAMERA POSE PERFORMANCE")
    print("=" * 60)
    print(f"Pose method:            {method}")

    print(f"\nTest pairs:             {total}")
    print(f"Successful poses:       {len(t)}")

    failure_label = (
        "RANSAC failures:" if method == "RANSAC"
        else "Prediction failures:"
    )
    print(f"{failure_label:24s}{failures}")
    print(f"Success rate:           {len(t) / max(total, 1):.3f}")
    print(f"Excluded GT pairs:      {excluded_gt_pairs}")

    print("\nTRANSLATION DIRECTION ERROR")
    print("-" * 40)
    print(f"Mean:                   {stat(t, np.mean):.2f}°")
    print(f"Median:                 {stat(t, np.median):.2f}°")
    print(
        f"P90:                    "
        f"{stat(t, lambda x: np.percentile(x, 90)):.2f}°"
    )

    print("\nYAW ERROR")
    print("-" * 40)
    print(f"Mean:                   {stat(yaw, np.mean):.2f}°")
    print(f"Median:                 {stat(yaw, np.median):.2f}°")
    print(
        f"P90:                    "
        f"{stat(yaw, lambda x: np.percentile(x, 90)):.2f}°"
    )

    print("\nACCURACY THRESHOLDS")
    print("-" * 40)

    for threshold in [5, 10, 20, 30]:
        t_acc = stat(t, lambda x: np.mean(x < threshold))
        yaw_acc = stat(yaw, lambda x: np.mean(x < threshold))

        print(
            f"< {threshold:2d}°: "
            f"T={100*t_acc:5.1f}% | "
            f"Yaw={100*yaw_acc:5.1f}%"
        )

    return {
        "translation_errors": t,
        "yaw_errors": yaw,
        "failures": failures,
        "total": total,
        "excluded_gt_pairs": excluded_gt_pairs,
        "method": method,

        "translation_mean": stat(t, np.mean),
        "translation_median": stat(t, np.median),
        "translation_p90": stat(
            t, lambda x: np.percentile(x, 90)
        ),

        "yaw_mean": stat(yaw, np.mean),
        "yaw_median": stat(yaw, np.median),
        "yaw_p90": stat(
            yaw, lambda x: np.percentile(x, 90)
        ),
    }