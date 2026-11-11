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
    # Occupancy F1 with a one-bin tolerance. Match detections one to one
    # within each view, so a single prediction cannot explain multiple GTs.
    # --------------------------------------------------

    def occupancy_counts_with_tolerance(gt, pred, threshold, tol):
        gt_binary = gt > threshold
        pred_binary = pred > threshold
        tp_tol = 0

        for gt_row, pred_row in zip(gt_binary, pred_binary):
            gt_indices = np.flatnonzero(gt_row)
            pred_indices = np.flatnonzero(pred_row)
            # Maximum-cardinality bipartite matching (augmenting paths).
            matched_gt_for_pred = {}

            def assign(pred_index, seen):
                for gt_index in gt_indices:
                    if abs(int(gt_index) - int(pred_index)) > tol or gt_index in seen:
                        continue
                    seen.add(gt_index)
                    if (gt_index not in matched_gt_for_pred
                            or assign(matched_gt_for_pred[gt_index], seen)):
                        matched_gt_for_pred[gt_index] = pred_index
                        return True
                return False

            for pred_index in pred_indices:
                assign(pred_index, set())
            tp_tol += len(matched_gt_for_pred)

        fp_tol = int(pred_binary.sum()) - tp_tol
        fn_tol = int(gt_binary.sum()) - tp_tol
        return tp_tol, fp_tol, fn_tol

    tp_1bin, fp_1bin, fn_1bin = occupancy_counts_with_tolerance(
        occ_gt, occ_pred, occ_threshold, 1
    )
    precision_1bin = tp_1bin / max(tp_1bin + fp_1bin, 1)
    recall_1bin = tp_1bin / max(tp_1bin + fn_1bin, 1)
    f1_1bin = 2 * precision_1bin * recall_1bin / max(
        precision_1bin + recall_1bin, 1e-12
    )
    f1_1bin_delta = f1_1bin - f1

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
        f"{'F1 ±1 bin':>12}"
    )
    print(
        f"{occ_accuracy:10.3f}"
        f"{precision:12.3f}"
        f"{recall:10.3f}"
        f"{f1:10.3f}"
        f"{f1_1bin:12.3f}"
    )


    print("\nGEOMETRY")
    print("-" * 78)
    print(
        f"{'':<10}"
        f"{'MAE':>10}"
        f"{'RMSE':>10}"
        f"{'Bias':>10}"
        f"{'Corr':>10}"
        f"{'Pred std':>10}"
        f"{'GT std':>10}"
    )

    print(
        f"{'Radius':<10}"
        f"{radius_mae:10.3f}"
        f"{radius_rmse:10.3f}"
        f"{radius_bias:10.3f}"
        f"{radius_corr:10.3f}"
        f"{radius_pred.std():10.3f}"
        f"{radius_gt.std():10.3f}"
    )

    print(
        f"{'Depth':<10}"
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
    )

    print(
        f"{'':<10}"
        f"{depth_05m:12.3f}"
        f"{depth_1m:12.3f}"
        f"{depth_2m:12.3f}"
    )

    return {
        "occupancy_accuracy": occ_accuracy,
        "occupancy_precision": precision,
        "occupancy_recall": recall,
        "occupancy_f1": f1,
        "occupancy_precision_1bin": precision_1bin,
        "occupancy_recall_1bin": recall_1bin,
        "occupancy_f1_1bin": f1_1bin,
        "occupancy_f1_1bin_delta": f1_1bin_delta,

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
def evaluate_pose(
    model,
    loader,
    device,
    gt_occ_threshold=0.5,
    matching_mode="predicted",
    pred_occ_threshold=0.2,
):
    """
    matching_mode:
        "gt"        : GT-bin och GT-matchningar.
        "gt_bins"   : GT-bin, men skattade matchningar.
        "predicted" : Skattade bin och skattade matchningar.

    Radie och djup är förutsagda i alla tre testerna.
    Avsedd för den uppdaterade modellen från model.py.
    """
    from model import predicted_bin_matches, pack_matched_pairs

    methods = {
        "gt": "Learned pose with GT bins and GT matches",
        "gt_bins": "Learned pose with GT bins and predicted matches",
        "predicted": "Learned pose with predicted bins and matches",
    }
    if matching_mode not in methods:
        raise ValueError(
            "matching_mode måste vara 'gt', 'gt_bins' eller 'predicted'."
        )

    model.eval()
    base_model = getattr(model, "_orig_mod", model)
    method = methods[matching_mode]

    translation_errors = []
    yaw_errors = []

    seen_pairs = 0
    total = 0
    failures = 0
    excluded_gt_pairs = 0

    for batch in loader:
        img_a, vision_a, img_b, vision_b, pose_gt = [
            x.to(device, non_blocking=True)
            for x in batch
        ]
        pose_gt = pose_gt.float()
        seen_pairs += len(pose_gt)

        # Endast GT-kameraposen avgör om felet går att beräkna.
        eligible = (
            torch.isfinite(pose_gt).all(dim=-1)
            & (pose_gt[:, :2].norm(dim=-1) > 1e-6)
            & (pose_gt[:, 2:].norm(dim=-1) > 1e-8)
        )

        excluded_gt_pairs += (~eligible).sum().item()
        total += eligible.sum().item()

        if not eligible.any():
            continue

        img_a = img_a[eligible]
        img_b = img_b[eligible]
        vision_a = vision_a[eligible]
        vision_b = vision_b[eligible]
        pose_gt = pose_gt[eligible]

        if matching_mode == "gt":
            # Explicit GT-läge behövs eftersom auto använder
            # skattade matchningar när modellen är i eval().
            _, _, pose_pred, pose_info = model(
                img_a,
                img_b,
                matching_mode="gt",
                target_vision_a=vision_a,
                target_vision_b=vision_b,
                occ_thresh=gt_occ_threshold,
                return_pose_inputs=True,
            )
            pose_valid = pose_info["pose_valid"]

        elif matching_mode == "predicted":
            # Ingen GT-cylinderdata skickas till modellen.
            _, _, pose_pred, pose_info = model(
                img_a,
                img_b,
                matching_mode="predicted",
                pred_occ_thresh=pred_occ_threshold,
                return_pose_inputs=True,
            )
            pose_valid = pose_info["pose_valid"]

        else:
            # GT-bin, men modellen får själv skatta kopplingarna.
            pred_a, pred_b, _, corr_a, corr_b = model(
                img_a,
                img_b,
                compute_pose=False,
                return_corr=True,
            )

            # Matcharen får GT-occupancy som kandidaturval.
            # Radie och djup behålls från prediktionerna.
            selection_a = pred_a.clone()
            selection_b = pred_b.clone()
            selection_a[..., 0] = vision_a[..., 0]
            selection_b[..., 0] = vision_b[..., 0]

            matches = predicted_bin_matches(
                selection_a,
                selection_b,
                corr_a,
                corr_b,
                fov_degrees=base_model.fov_degrees,
                occ_thresh=gt_occ_threshold,
                # Behåll alla GT-kandidatbin utan NMS eller top-k.
                nms_radius=0,
                max_candidates=max(
                    2, vision_a.shape[1], vision_b.shape[1]
                ),
            )

            # Posehuvudet får originalprediktionernas geometri.
            tokens, pair_mask, _ = pack_matched_pairs(
                pred_a,
                pred_b,
                matches,
                base_model.fov_degrees,
            )
            pose_valid = pair_mask.sum(dim=1) >= 2

            safe_mask = pair_mask.clone()
            safe_mask[~safe_mask.any(dim=1), 0] = True

            pose_pred = base_model.pose_head(tokens, safe_mask)
            pose_pred = torch.where(
                pose_valid.unsqueeze(-1),
                pose_pred,
                torch.zeros_like(pose_pred),
            )

        if pose_pred is None:
            raise ValueError("Modellen returnerade ingen pose.")

        pose_pred = pose_pred.float()

        valid = (
            pose_valid
            & torch.isfinite(pose_pred).all(dim=-1)
            & (pose_pred[:, :2].norm(dim=-1) >= 1e-8)
            & (pose_pred[:, 2:].norm(dim=-1) >= 1e-8)
        )

        # För få matchningar är ett prediktionsfel,
        # inte ett ogiltigt GT-par.
        failures += (~valid).sum().item()

        pred = pose_pred[valid]
        gt = pose_gt[valid]

        t_pred = F.normalize(pred[:, :2], dim=-1)
        t_gt = F.normalize(gt[:, :2], dim=-1)

        cos_t = (t_pred * t_gt).sum(dim=-1).clamp(-1.0, 1.0)
        translation_errors.extend(
            torch.rad2deg(torch.acos(cos_t)).cpu().tolist()
        )

        yaw_pred = torch.atan2(pred[:, 2], pred[:, 3])
        yaw_gt = torch.atan2(gt[:, 2], gt[:, 3])
        delta_yaw = yaw_pred - yaw_gt

        yaw_errors.extend(
            torch.rad2deg(
                torch.atan2(
                    torch.sin(delta_yaw),
                    torch.cos(delta_yaw),
                )
            ).abs().cpu().tolist()
        )

    t = np.asarray(translation_errors, dtype=float)
    yaw = np.asarray(yaw_errors, dtype=float)

    def stat(values, function):
        return float(function(values)) if len(values) else float("nan")

    metrics = {
        "translation_errors": t,
        "yaw_errors": yaw,
        "seen_pairs": seen_pairs,
        "failures": failures,
        "total": total,
        "excluded_gt_pairs": excluded_gt_pairs,
        "method": method,
        "matching_mode": matching_mode,
        "success_rate": len(t) / total if total else float("nan"),
    }

    for name, values in (("translation", t), ("yaw", yaw)):
        metrics[name + "_mean"] = stat(values, np.mean)
        metrics[name + "_median"] = stat(values, np.median)
        metrics[name + "_p90"] = stat(
            values, lambda x: np.percentile(x, 90)
        )

    print("CAMERA POSE PERFORMANCE")
    print("=" * 60)
    print(f"Pose method:            {method}")
    print(f"\nSeen pairs:             {seen_pairs}")
    print(f"Test pairs:             {total}")
    print(f"Successful poses:       {len(t)}")
    print(f"Prediction failures:    {failures}")
    print(f"Success rate:           {metrics['success_rate']:.3f}")
    print(f"Excluded GT pairs:      {excluded_gt_pairs}")

    for title, name in (
        ("TRANSLATION DIRECTION ERROR", "translation"),
        ("YAW ERROR", "yaw"),
    ):
        print("\n" + title)
        print("-" * 40)
        print(f"Mean:                   {metrics[name + '_mean']:.2f}°")
        print(f"Median:                 {metrics[name + '_median']:.2f}°")
        print(f"P90:                    {metrics[name + '_p90']:.2f}°")

    print("\nACCURACY THRESHOLDS — successful poses only")
    print("-" * 40)

    for threshold in [5, 10, 20, 30]:
        t_acc = stat(t, lambda x: np.mean(x < threshold))
        yaw_acc = stat(yaw, lambda x: np.mean(x < threshold))

        print(
            f"< {threshold:2d}°: "
            f"T={100*t_acc:5.1f}% | "
            f"Yaw={100*yaw_acc:5.1f}%"
        )

    return metrics

@torch.no_grad()
def evaluate_pose_forest(model, loader, device, predict_pose=None):
    """
    loader returnerar (img_a, img_b, T_ab), där T_ab är [B, 4, 4]
    och mappar kamera A -> kamera B.

    Förutsagd pose: [forward, right, sin(yaw), cos(yaw)].

    predict_pose är valfri:
        predict_pose(model, img_a, img_b) -> (pose, pose_valid)

    pose_valid: bool-tensor [B]. Ogiltiga prediktioner räknas som
    failures, även bildpar med för få matchningar.
    """
    model.eval()
    base_model = getattr(model, "_orig_mod", model)
    parameters = inspect.signature(base_model.forward).parameters

    if predict_pose is not None:
        method = "Custom forest pose prediction"
    elif callable(getattr(base_model, "predict_pose_ransac", None)):
        method = "RANSAC"
    elif all(
        name in parameters
        for name in (
            "target_vision_a",
            "target_vision_b",
            "return_pose_inputs",
        )
    ):
        raise ValueError(
            "Modellen kräver cylinderbinmatchningar för pose. "
            "Forest-datan innehåller endast kamerornas relativa poser. "
            "Ange predict_pose som beräknar pose från bilderna med "
            "oberoende skattade matchningar."
        )
    else:
        method = "Direct learned pose"

    translation_errors = []
    yaw_errors = []
    total = 0
    failures = 0
    excluded_gt_pairs = 0

    for batch in loader:
        img_a, img_b, target_T = [
            x.to(device, non_blocking=True)
            for x in batch
        ]
        target_T = target_T.float()

        if target_T.ndim != 3 or target_T.shape[1:] != (4, 4):
            raise ValueError("Forest GT måste ha format [B, 4, 4].")

        rotation = target_T[:, :3, :3]

        # Samma omvandling som relative_pose_loss_forest:
        # translation = [framåt (Z), höger (X)].
        pose_gt = torch.stack(
            [
                target_T[:, 2, 3],
                target_T[:, 0, 3],
                0.5 * (rotation[:, 0, 2] - rotation[:, 2, 0]),
                0.5 * (rotation[:, 0, 0] + rotation[:, 2, 2]),
            ],
            dim=-1,
        )

        eligible = (
            torch.isfinite(target_T).all(dim=-1).all(dim=-1)
            & (pose_gt[:, :2].norm(dim=-1) > 1e-6)
            & (pose_gt[:, 2:].norm(dim=-1) > 1e-8)
        )

        excluded_gt_pairs += (~eligible).sum().item()
        total += eligible.sum().item()

        if not eligible.any():
            continue

        img_a = img_a[eligible]
        img_b = img_b[eligible]
        pose_gt = pose_gt[eligible]

        if predict_pose is not None:
            pose_pred, pose_valid = predict_pose(model, img_a, img_b)
        else:
            if method == "RANSAC":
                _, _, pose_pred = base_model.predict_pose_ransac(
                    img_a, img_b
                )
            else:
                _, _, pose_pred = model(img_a, img_b)

            pose_valid = torch.ones(
                len(pose_gt), dtype=torch.bool, device=device
            )

        if pose_pred is None or pose_pred.shape != pose_gt.shape:
            raise ValueError("Förutsagd pose måste ha format [B, 4].")

        pose_pred = pose_pred.to(device=device, dtype=torch.float32)
        pose_valid = torch.as_tensor(pose_valid, device=device)

        if (
            pose_valid.dtype != torch.bool
            or pose_valid.shape != (len(pose_gt),)
        ):
            raise ValueError("pose_valid måste vara en bool-tensor [B].")

        valid = (
            pose_valid
            & torch.isfinite(pose_pred).all(dim=-1)
            & (pose_pred[:, :2].norm(dim=-1) >= 1e-8)
            & (pose_pred[:, 2:].norm(dim=-1) >= 1e-8)
        )

        failures += (~valid).sum().item()
        pred = pose_pred[valid]
        gt = pose_gt[valid]

        t_pred = F.normalize(pred[:, :2], dim=-1)
        t_gt = F.normalize(gt[:, :2], dim=-1)
        cos_t = (t_pred * t_gt).sum(dim=-1).clamp(-1.0, 1.0)

        translation_errors.extend(
            torch.rad2deg(torch.acos(cos_t)).cpu().tolist()
        )

        yaw_pred = torch.atan2(pred[:, 2], pred[:, 3])
        yaw_gt = torch.atan2(gt[:, 2], gt[:, 3])
        delta = yaw_pred - yaw_gt

        yaw_errors.extend(
            torch.rad2deg(
                torch.atan2(torch.sin(delta), torch.cos(delta))
            ).abs().cpu().tolist()
        )

    t = np.asarray(translation_errors, dtype=float)
    yaw = np.asarray(yaw_errors, dtype=float)

    def stat(values, function):
        return float(function(values)) if len(values) else float("nan")

    metrics = {
        "translation_errors": t,
        "yaw_errors": yaw,
        "failures": failures,
        "total": total,
        "excluded_gt_pairs": excluded_gt_pairs,
        "method": method,
        "success_rate": len(t) / total if total else float("nan"),
    }

    for name, values in (("translation", t), ("yaw", yaw)):
        metrics[name + "_mean"] = stat(values, np.mean)
        metrics[name + "_median"] = stat(values, np.median)
        metrics[name + "_p90"] = stat(
            values, lambda x: np.percentile(x, 90)
        )

    print("FOREST CAMERA POSE PERFORMANCE")
    print("=" * 60)
    print(f"Pose method:            {method}")
    print(f"\nTest pairs:             {total}")
    print(f"Successful poses:       {len(t)}")
    print(f"Prediction failures:    {failures}")
    print(f"Success rate:           {metrics['success_rate']:.3f}")
    print(f"Excluded GT pairs:      {excluded_gt_pairs}")

    for title, name in (
        ("TRANSLATION DIRECTION ERROR", "translation"),
        ("YAW ERROR", "yaw"),
    ):
        print("\n" + title)
        print("-" * 40)
        print(f"Mean:                   {metrics[name + '_mean']:.2f}°")
        print(f"Median:                 {metrics[name + '_median']:.2f}°")
        print(f"P90:                    {metrics[name + '_p90']:.2f}°")

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

    return metrics
