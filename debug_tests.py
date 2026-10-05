import torch
import torch.nn.functional as F
import numpy as np



def test_matched_depth_consistency(model, loader, device, occ_thresh=0.5):
    model.eval()
    err_a, err_b, pred_delta, gt_delta = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            batch = [x.to(device, non_blocking=True) for x in batch]
            pairs = [(batch[0], batch[1], batch[2], batch[3])] if len(batch) == 5 else [
                (batch[0], batch[1], batch[2], batch[3]),
                (batch[5], batch[6], batch[7], batch[8]),
            ]

            for img_a, gt_a, img_b, gt_b in pairs:
                pred_a, pred_b, _ = model(img_a, img_b)

                for b in range(img_a.shape[0]):
                    occ_a = gt_a[b, :, 0] > occ_thresh
                    occ_b = gt_b[b, :, 0] > occ_thresh
                    ids_a = gt_a[b, :, 3].round().long()
                    ids_b = gt_b[b, :, 3].round().long()

                    for cid in torch.unique(ids_a[occ_a]):
                        ia = torch.where(occ_a & (ids_a == cid))[0]
                        ib = torch.where(occ_b & (ids_b == cid))[0]
                        if ia.numel() != 1 or ib.numel() != 1:
                            continue

                        pa, pb = pred_a[b, ia[0], 2].item(), pred_b[b, ib[0], 2].item()
                        ga, gb = gt_a[b, ia[0], 2].item(), gt_b[b, ib[0], 2].item()

                        err_a.append(pa - ga)
                        err_b.append(pb - gb)
                        pred_delta.append(pb - pa)
                        gt_delta.append(gb - ga)

    err_a, err_b = np.array(err_a), np.array(err_b)
    pred_delta, gt_delta = np.array(pred_delta), np.array(gt_delta)
    error_diff = np.abs(err_a - err_b)

    print("MATCHED DEPTH CONSISTENCY")
    print("-" * 50)
    print(f"Matched cylinders:            {len(err_a)}")
    print(f"A MAE:                        {np.mean(np.abs(err_a)):.3f}")
    print(f"B MAE:                        {np.mean(np.abs(err_b)):.3f}")
    print(f"Corr error A vs B:            {np.corrcoef(err_a, err_b)[0,1]:.3f}")
    print(f"Mean |error A - error B|:     {error_diff.mean():.3f}")
    print(f"Median |error A - error B|:   {np.median(error_diff):.3f}")
    print(f"GT mean |delta depth|:        {np.mean(np.abs(gt_delta)):.3f}")
    print(f"Pred mean |delta depth|:      {np.mean(np.abs(pred_delta)):.3f}")
    print(f"Mean delta error:             {np.mean(np.abs(pred_delta - gt_delta)):.3f}")
    print(f"Corr pred vs GT delta:        {np.corrcoef(pred_delta, gt_delta)[0,1]:.3f}")

    return {
        "errors_a": err_a,
        "errors_b": err_b,
        "pred_delta": pred_delta,
        "gt_delta": gt_delta,
    }


def test_new_gt_pose_geometry(loader, device, fov_degrees=90.0):
    errors = []

    with torch.no_grad():
        for batch in loader:
            batch = [x.to(device) for x in batch]

            pairs = [(batch[1], batch[3], batch[4])] if len(batch) == 5 else [
                (batch[1], batch[3], batch[4]),
                (batch[6], batch[8], batch[9]),
            ]

            for va, vb, pose in pairs:
                B, N, _ = va.shape
                fov = np.deg2rad(fov_degrees)

                theta = torch.linspace(
                    -0.5 * fov, 0.5 * fov, N,
                    device=device, dtype=va.dtype,
                )

                # A-punkter i Cam1-frame
                pa = torch.stack([
                    va[..., 2] * torch.cos(theta),
                    va[..., 2] * torch.sin(theta),
                ], dim=-1)

                # B-punkter i Cam2-frame
                pb = torch.stack([
                    vb[..., 2] * torch.cos(theta),
                    vb[..., 2] * torch.sin(theta),
                ], dim=-1)

                for b in range(B):
                    occ_a = va[b, :, 0] > 0.5
                    occ_b = vb[b, :, 0] > 0.5
                    ids_a = va[b, :, 3].round().long()
                    ids_b = vb[b, :, 3].round().long()

                    t = pose[b, :2]

                    yaw = F.normalize(pose[b, 2:], dim=0)
                    s, c = yaw[0], yaw[1]

                    # Cam1/world -> Cam2 orientation
                    R_inv = torch.stack([
                        torch.stack([ c,  s]),
                        torch.stack([-s,  c]),
                    ])

                    for cid in torch.unique(ids_a[occ_a]):
                        ia = torch.where(occ_a & (ids_a == cid))[0]
                        ib = torch.where(occ_b & (ids_b == cid))[0]

                        if ia.numel() != 1 or ib.numel() != 1:
                            continue

                        A = pa[b, ia[0]]
                        B_gt = pb[b, ib[0]]

                        # Cam2 ligger vid t i Cam1-frame:
                        # flytta först origo till Cam2, rotera sedan.
                        B_pred = R_inv @ (A - t)

                        errors.append(
                            torch.linalg.vector_norm(
                                B_pred - B_gt
                            ).item()
                        )

    errors = np.asarray(errors)

    print("NEW GT POSE GEOMETRY TEST")
    print("-" * 45)
    print(f"Matches: {len(errors)}")
    print(f"Mean:    {errors.mean():.4f} m")
    print(f"Median:  {np.median(errors):.4f} m")
    print(f"P90:     {np.percentile(errors, 90):.4f} m")
    print(f"Max:     {errors.max():.4f} m")

    return errors

def evaluate_train_geometry(model, loader, device, occ_thresh=0.5):
    model.eval()

    pred_radius_all, gt_radius_all = [], []
    pred_depth_all, gt_depth_all = [], []

    radius_pred_mae, radius_base_mae = [], []
    depth_pred_mae, depth_base_mae = [], []
    radius_pred_std, radius_gt_std = [], []
    depth_pred_std, depth_gt_std = [], []
    radius_sample_corr, depth_sample_corr = [], []

    with torch.no_grad():
        for batch in loader:
            batch = [x.to(device, non_blocking=True) for x in batch]

            if len(batch) == 10:
                pairs = [
                    (batch[0], batch[1], batch[2], batch[3]),
                    (batch[5], batch[6], batch[7], batch[8]),
                ]
            elif len(batch) == 5:
                pairs = [(batch[0], batch[1], batch[2], batch[3])]
            else:
                raise ValueError(f"Unexpected batch length: {len(batch)}")

            for img_a, vision_a, img_b, vision_b in pairs:
                pred_a, pred_b, _ = model(img_a, img_b)

                for pred, gt in [(pred_a, vision_a), (pred_b, vision_b)]:
                    for b in range(pred.shape[0]):
                        mask = gt[b, :, 0] > occ_thresh

                        if mask.sum() < 2:
                            continue

                        pr = pred[b, mask, 1].float()
                        gr = gt[b, mask, 1].float()
                        pd = pred[b, mask, 2].float()
                        gd = gt[b, mask, 2].float()

                        pred_radius_all.append(pr.cpu())
                        gt_radius_all.append(gr.cpu())
                        pred_depth_all.append(pd.cpu())
                        gt_depth_all.append(gd.cpu())

                        radius_pred_mae.append(torch.mean(torch.abs(pr - gr)).item())
                        depth_pred_mae.append(torch.mean(torch.abs(pd - gd)).item())

                        radius_base_mae.append(torch.mean(torch.abs(gr.mean() - gr)).item())
                        depth_base_mae.append(torch.mean(torch.abs(gd.mean() - gd)).item())

                        radius_pred_std.append(pr.std().item())
                        radius_gt_std.append(gr.std().item())
                        depth_pred_std.append(pd.std().item())
                        depth_gt_std.append(gd.std().item())

                        if pr.std() > 1e-6 and gr.std() > 1e-6:
                            r = torch.corrcoef(torch.stack([pr, gr]))[0, 1]
                            if torch.isfinite(r):
                                radius_sample_corr.append(r.item())

                        if pd.std() > 1e-6 and gd.std() > 1e-6:
                            r = torch.corrcoef(torch.stack([pd, gd]))[0, 1]
                            if torch.isfinite(r):
                                depth_sample_corr.append(r.item())

    pred_radius = torch.cat(pred_radius_all).numpy()
    gt_radius = torch.cat(gt_radius_all).numpy()
    pred_depth = torch.cat(pred_depth_all).numpy()
    gt_depth = torch.cat(gt_depth_all).numpy()

    print("TRAIN GEOMETRY")
    print("=" * 50)

    print("\nRADIUS")
    print("-" * 50)
    print(f"Global MAE:               {np.mean(np.abs(pred_radius - gt_radius)):.4f}")
    print(f"Global correlation:       {np.corrcoef(pred_radius, gt_radius)[0,1]:.3f}")
    print(f"Model MAE/sample:         {np.mean(radius_pred_mae):.4f}")
    print(f"Mean-baseline MAE:        {np.mean(radius_base_mae):.4f}")
    print(f"Model / baseline:         {np.mean(radius_pred_mae) / np.mean(radius_base_mae):.3f}x")
    print(f"Pred std across bins:     {np.mean(radius_pred_std):.4f}")
    print(f"GT std across bins:       {np.mean(radius_gt_std):.4f}")
    print(f"Mean per-sample corr:     {np.mean(radius_sample_corr):.3f}")
    print(f"Median per-sample corr:   {np.median(radius_sample_corr):.3f}")

    print("\nDEPTH")
    print("-" * 50)
    print(f"Global MAE:               {np.mean(np.abs(pred_depth - gt_depth)):.4f}")
    print(f"Global correlation:       {np.corrcoef(pred_depth, gt_depth)[0,1]:.3f}")
    print(f"Model MAE/sample:         {np.mean(depth_pred_mae):.4f}")
    print(f"Mean-baseline MAE:        {np.mean(depth_base_mae):.4f}")
    print(f"Model / baseline:         {np.mean(depth_pred_mae) / np.mean(depth_base_mae):.3f}x")
    print(f"Pred std across bins:     {np.mean(depth_pred_std):.4f}")
    print(f"GT std across bins:       {np.mean(depth_gt_std):.4f}")
    print(f"Mean per-sample corr:     {np.mean(depth_sample_corr):.3f}")
    print(f"Median per-sample corr:   {np.median(depth_sample_corr):.3f}")


def test_depth_by_distance(model, loader, device, bins=(0, 5, 10, 15, 20, 30, float("inf"))):
    model.eval()
    data = {i: {"pred": [], "gt": []} for i in range(len(bins) - 1)}

    with torch.no_grad():
        for batch in loader:
            batch = [x.to(device, non_blocking=True) for x in batch]
            pairs = [(batch[0], batch[1], batch[2], batch[3])] if len(batch) == 5 else [
                (batch[0], batch[1], batch[2], batch[3]),
                (batch[5], batch[6], batch[7], batch[8]),
            ]

            for img_a, gt_a, img_b, gt_b in pairs:
                pred_a, pred_b, _ = model(img_a, img_b)

                for pred, gt in [(pred_a, gt_a), (pred_b, gt_b)]:
                    mask = gt[..., 0] > 0.5
                    p = pred[..., 2][mask].cpu().numpy()
                    g = gt[..., 2][mask].cpu().numpy()

                    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
                        m = (g >= lo) & (g < hi)
                        data[i]["pred"].extend(p[m])
                        data[i]["gt"].extend(g[m])

    print("DEPTH ERROR BY GT DISTANCE")
    print("-" * 82)
    print(
        f"{'Range':>10} {'N':>7} {'MAE':>8} {'Rel MAE':>9} "
        f"{'Corr':>8} {'Pred mean':>10} {'GT mean':>9} "
        f"{'Pred std':>10} {'GT std':>9}"
    )
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        p = np.asarray(data[i]["pred"])
        g = np.asarray(data[i]["gt"])

        if len(g) == 0:
            continue

        mae = np.mean(np.abs(p - g))
        rel = np.mean(np.abs(p - g) / np.maximum(g, 1e-6))
        corr = np.corrcoef(p, g)[0, 1] if len(g) > 1 and p.std() > 1e-8 and g.std() > 1e-8 else np.nan

        label = f"{lo:g}-{hi:g}" if np.isfinite(hi) else f"{lo:g}+"

        print(
            f"{label:>10} {len(g):7d} {mae:8.3f} {rel:9.3f} "
            f"{corr:8.3f} {p.mean():10.3f} {g.mean():9.3f} "
            f"{p.std():10.3f} {g.std():9.3f}"
        )

    return data


def evaluate_16_correspondences(model, loader, device, occ_thresh=0.5, temperature=0.1, fov_degrees=90.0):
    model.eval()
    total = exact = within_1 = within_2 = 0
    errors = []

    with torch.no_grad():
        for batch in loader:
            batch = [x.to(device, non_blocking=True) for x in batch]
            pairs = [(batch[0], batch[1], batch[2], batch[3])] if len(batch) == 5 else [
                (batch[0], batch[1], batch[2], batch[3]),
                (batch[5], batch[6], batch[7], batch[8]),
            ]

            for img_a, vision_a, img_b, vision_b in pairs:
                _, _, _, corr_a, corr_b = model(img_a, img_b, return_corr=True)
                corr_a = F.normalize(corr_a, dim=-1)
                corr_b = F.normalize(corr_b, dim=-1)

                fov = torch.tensor(fov_degrees * torch.pi / 180.0, device=device, dtype=corr_a.dtype)

                def bin_to_col(idx, num_bins):
                    theta = -0.5 * fov + idx.to(corr_a.dtype) / (num_bins - 1) * fov
                    x = 0.5 - 0.5 * torch.tan(theta) / torch.tan(0.5 * fov)
                    return torch.clamp((x * 16).long(), 0, 15)

                for b in range(img_a.shape[0]):
                    idx_a = torch.where(vision_a[b, :, 0] > occ_thresh)[0]
                    idx_b = torch.where(vision_b[b, :, 0] > occ_thresh)[0]
                    ids_a = vision_a[b, idx_a, 3].round().long()
                    ids_b = vision_b[b, idx_b, 3].round().long()

                    sim = corr_a[b] @ corr_b[b].T / temperature
                    pred_b = sim.argmax(dim=-1)

                    for cid in ids_a.unique():
                        ma = idx_a[ids_a == cid]
                        mb = idx_b[ids_b == cid]

                        if ma.numel() != 1 or mb.numel() != 1:
                            continue

                        ca = bin_to_col(ma[0], vision_a.shape[1]).item()
                        cb = bin_to_col(mb[0], vision_b.shape[1]).item()

                        error = abs(pred_b[ca].item() - cb)

                        total += 1
                        exact += error == 0
                        within_1 += error <= 1
                        within_2 += error <= 2
                        errors.append(error)

    print("16-COLUMN CORRESPONDENCE")
    print("----------------------------------------")
    print(f"Matches:       {total}")
    print(f"Exact:         {100 * exact / total:.1f}%")
    print(f"Within ±1:     {100 * within_1 / total:.1f}%")
    print(f"Within ±2:     {100 * within_2 / total:.1f}%")
    print(f"Mean error:    {np.mean(errors):.2f} columns")
    print(f"Median error:  {np.median(errors):.2f} columns")

def evaluate_16_soft_shift(model, loader, device, occ_thresh=0.5, temperature=0.1, fov_degrees=90.0):
    model.eval()
    pred_shifts = []
    gt_shifts = []

    with torch.no_grad():
        for batch in loader:
            batch = [x.to(device, non_blocking=True) for x in batch]
            pairs = [(batch[0], batch[1], batch[2], batch[3])] if len(batch) == 5 else [
                (batch[0], batch[1], batch[2], batch[3]),
                (batch[5], batch[6], batch[7], batch[8]),
            ]

            for img_a, vision_a, img_b, vision_b in pairs:
                _, _, _, corr_a, corr_b = model(img_a, img_b, return_corr=True)
                corr_a = F.normalize(corr_a, dim=-1)
                corr_b = F.normalize(corr_b, dim=-1)

                prob = F.softmax(torch.matmul(corr_a, corr_b.transpose(1, 2)) / temperature, dim=-1)

                # Continuous column coordinates: 0 ... 15
                coords = torch.arange(16, device=device, dtype=prob.dtype)
                expected_b = torch.matmul(prob, coords)

                fov = torch.tensor(fov_degrees * torch.pi / 180.0, device=device, dtype=prob.dtype)

                def bin_to_col_float(idx, num_bins):
                    theta = -0.5 * fov + idx.to(prob.dtype) / (num_bins - 1) * fov
                    x = 0.5 - 0.5 * torch.tan(theta) / torch.tan(0.5 * fov)
                    return torch.clamp(x * 16.0 - 0.5, 0.0, 15.0)

                for b in range(img_a.shape[0]):
                    idx_a = torch.where(vision_a[b, :, 0] > occ_thresh)[0]
                    idx_b = torch.where(vision_b[b, :, 0] > occ_thresh)[0]
                    ids_a = vision_a[b, idx_a, 3].round().long()
                    ids_b = vision_b[b, idx_b, 3].round().long()

                    for cid in ids_a.unique():
                        ma = idx_a[ids_a == cid]
                        mb = idx_b[ids_b == cid]

                        if ma.numel() != 1 or mb.numel() != 1:
                            continue

                        ca = bin_to_col_float(ma[0], vision_a.shape[1])
                        cb = bin_to_col_float(mb[0], vision_b.shape[1])

                        # Descriptor row corresponding to nearest A column
                        ca_idx = torch.clamp(torch.round(ca).long(), 0, 15)

                        pred_shifts.append((expected_b[b, ca_idx] - ca).item())
                        gt_shifts.append((cb - ca).item())

    pred_shifts = np.array(pred_shifts)
    gt_shifts = np.array(gt_shifts)

    print("16-COLUMN SOFT SHIFT")
    print("----------------------------------------")
    print(f"Matches:                  {len(pred_shifts)}")
    print(f"Mean |pred shift|:        {np.mean(np.abs(pred_shifts)):.2f} columns")
    print(f"Mean |GT shift|:          {np.mean(np.abs(gt_shifts)):.2f} columns")
    print(f"Mean shift error:         {np.mean(np.abs(pred_shifts - gt_shifts)):.2f} columns")
    print(f"Median shift error:       {np.median(np.abs(pred_shifts - gt_shifts)):.2f} columns")
    print(f"Correlation pred vs GT:   {np.corrcoef(pred_shifts, gt_shifts)[0,1]:.3f}")



def test_ransac_all_gt_new_pose(
    loader,
    device,
    fov_degrees=90.0,
    inlier_threshold=0.4,
    num_iters=200,
):
    t_errors, yaw_errors = [], []
    residuals_all = []
    match_counts, inlier_counts = [], []

    def rigid_transform(A, B):
        ca, cb = A.mean(0), B.mean(0)
        Ac, Bc = A - ca, B - cb

        dot = (Ac[:, 0] * Bc[:, 0] + Ac[:, 1] * Bc[:, 1]).sum()
        cross = (Ac[:, 0] * Bc[:, 1] - Ac[:, 1] * Bc[:, 0]).sum()

        norm = torch.sqrt(dot.square() + cross.square() + 1e-8)
        c, s = dot / norm, cross / norm

        R = torch.stack([
            torch.stack([c, -s]),
            torch.stack([s,  c]),
        ])

        t = cb - R @ ca
        return R, t, s, c

    def ransac(A, B):
        n = len(A)
        if n < 2:
            return None

        best = None
        best_count = 0
        best_error = float("inf")

        for _ in range(num_iters):
            idx = torch.randperm(n, device=A.device)[:2]
            R, t, _, _ = rigid_transform(A[idx], B[idx])

            pred = A @ R.T + t
            res = torch.linalg.vector_norm(pred - B, dim=-1)
            mask = res < inlier_threshold
            count = mask.sum().item()

            if count < 2:
                continue

            error = res[mask].mean().item()

            if count > best_count or (
                count == best_count and error < best_error
            ):
                best = mask
                best_count = count
                best_error = error

        if best is None:
            return None

        R, t, s, c = rigid_transform(A[best], B[best])
        return R, t, s, c, best

    with torch.no_grad():
        for batch in loader:
            batch = [x.to(device) for x in batch]

            pairs = (
                [(batch[1], batch[3], batch[4])]
                if len(batch) == 5
                else [
                    (batch[1], batch[3], batch[4]),
                    (batch[6], batch[8], batch[9]),
                ]
            )

            for va, vb, pose_gt in pairs:
                Bsz, N, _ = va.shape
                fov = torch.tensor(
                    np.deg2rad(fov_degrees),
                    device=device,
                    dtype=va.dtype,
                )

                theta = torch.linspace(
                    -0.5 * fov,
                    0.5 * fov,
                    N,
                    device=device,
                    dtype=va.dtype,
                )

                pts_a = torch.stack([
                    va[..., 2] * torch.cos(theta),
                    va[..., 2] * torch.sin(theta),
                ], -1)

                pts_b = torch.stack([
                    vb[..., 2] * torch.cos(theta),
                    vb[..., 2] * torch.sin(theta),
                ], -1)

                for b in range(Bsz):
                    occ_a = va[b, :, 0] > 0.5
                    occ_b = vb[b, :, 0] > 0.5

                    ids_a = va[b, :, 3].round().long()
                    ids_b = vb[b, :, 3].round().long()

                    A, Bpts = [], []

                    for cid in torch.unique(ids_a[occ_a]):
                        ia = torch.where(
                            occ_a & (ids_a == cid)
                        )[0]
                        ib = torch.where(
                            occ_b & (ids_b == cid)
                        )[0]

                        if ia.numel() != 1 or ib.numel() != 1:
                            continue

                        A.append(pts_a[b, ia[0]])
                        Bpts.append(pts_b[b, ib[0]])

                    if len(A) < 2:
                        continue

                    A = torch.stack(A)
                    Bpts = torch.stack(Bpts)

                    result = ransac(A, Bpts)
                    if result is None:
                        continue

                    R, t_ext, s_ext, c_ext, inliers = result

                    # RANSAC:
                    #     p_B = R @ p_A + t_ext
                    #
                    # GT pose:
                    #     Camera2 position/orientation in Camera1 frame
                    #
                    # Extrinsic -> camera pose:
                    t_pose = -R.T @ t_ext
                    s_pose = -s_ext
                    c_pose = c_ext

                    # Translation direction error
                    t_pred = F.normalize(t_pose, dim=0)
                    t_gt = F.normalize(pose_gt[b, :2], dim=0)

                    cos_t = torch.clamp(
                        torch.dot(t_pred, t_gt), -1.0, 1.0
                    )
                    t_err = torch.rad2deg(torch.acos(cos_t))

                    # Yaw error
                    yaw_gt = F.normalize(
                        pose_gt[b, 2:], dim=0
                    )

                    yaw_pred_angle = torch.atan2(
                        s_pose, c_pose
                    )
                    yaw_gt_angle = torch.atan2(
                        yaw_gt[0], yaw_gt[1]
                    )

                    dyaw = torch.atan2(
                        torch.sin(yaw_pred_angle - yaw_gt_angle),
                        torch.cos(yaw_pred_angle - yaw_gt_angle),
                    )

                    yaw_err = torch.abs(
                        torch.rad2deg(dyaw)
                    )

                    # Final geometric residual
                    pred_B = A @ R.T + t_ext
                    residual = torch.linalg.vector_norm(
                        pred_B - Bpts, dim=-1
                    )

                    t_errors.append(t_err.item())
                    yaw_errors.append(yaw_err.item())
                    residuals_all.extend(residual.cpu().tolist())
                    match_counts.append(len(A))
                    inlier_counts.append(inliers.sum().item())

    print("ALL-GT RANSAC — NEW POSE DEFINITION")
    print("-" * 55)
    print(f"Samples:              {len(t_errors)}")
    print(f"Mean matches/sample:  {np.mean(match_counts):.1f}")
    print(f"Mean inliers/sample:  {np.mean(inlier_counts):.1f}")
    print(f"Translation mean:     {np.mean(t_errors):.2f}°")
    print(f"Translation median:   {np.median(t_errors):.2f}°")
    print(f"Yaw mean:             {np.mean(yaw_errors):.2f}°")
    print(f"Yaw median:           {np.median(yaw_errors):.2f}°")
    print(f"Point residual mean:  {np.mean(residuals_all):.4f} m")
    print(f"Point residual median:{np.median(residuals_all):.4f} m")

    return {
        "translation_errors": t_errors,
        "yaw_errors": yaw_errors,
        "residuals": residuals_all,
    }

def test_ransac_geometry_ablation_gt_coarse(
    model,
    loader,
    device,
    occ_thresh=0.5,
    inlier_threshold=0.4,
    num_iters=200,
    fov_degrees=90.0,
):
    model.eval()

    configs = [
        ("All GT",              False, False, False),
        ("Pred depth only",     True,  False, False),
        ("Pred radius only",    False, True,  False),
        ("Pred occupancy only", False, False, True),
        ("All predicted",       True,  True,  True),
    ]

    results = {}

    # --------------------------------------------------
    # Rigid transform: p_B = R @ p_A + t_ext
    # --------------------------------------------------
    def rigid_transform(A, B):
        ca, cb = A.mean(0), B.mean(0)
        Ac, Bc = A - ca, B - cb

        dot = (Ac[:, 0] * Bc[:, 0] + Ac[:, 1] * Bc[:, 1]).sum()
        cross = (Ac[:, 0] * Bc[:, 1] - Ac[:, 1] * Bc[:, 0]).sum()

        norm = torch.sqrt(dot.square() + cross.square() + 1e-8)
        c, s = dot / norm, cross / norm

        R = torch.stack([
            torch.stack([c, -s]),
            torch.stack([s,  c]),
        ])

        t_ext = cb - R @ ca
        return R, t_ext, s, c

    def ransac(A, B):
        n = A.shape[0]
        if n < 2:
            return None

        best_mask = None
        best_count = 0
        best_error = float("inf")

        for _ in range(num_iters):
            idx = torch.randperm(n, device=A.device)[:2]
            R, t, _, _ = rigid_transform(A[idx], B[idx])

            pred = A @ R.T + t
            residuals = torch.linalg.vector_norm(pred - B, dim=-1)

            mask = residuals < inlier_threshold
            count = mask.sum().item()

            if count < 2:
                continue

            error = residuals[mask].mean().item()

            if count > best_count or (
                count == best_count and error < best_error
            ):
                best_mask = mask
                best_count = count
                best_error = error

        if best_mask is None:
            return None

        R, t, s, c = rigid_transform(
            A[best_mask],
            B[best_mask],
        )

        return R, t, s, c, best_mask

    # --------------------------------------------------
    # Evaluate one configuration
    # --------------------------------------------------
    def evaluate(use_pred_depth, use_pred_radius, use_pred_occ):
        t_errors, yaw_errors = [], []
        matches_all, inliers_all = [], []
        zeros = 0

        with torch.no_grad():
            for batch in loader:
                batch = [
                    x.to(device, non_blocking=True)
                    for x in batch
                ]

                pairs = (
                    [(batch[0], batch[1], batch[2], batch[3], batch[4])]
                    if len(batch) == 5
                    else [
                        (batch[0], batch[1], batch[2], batch[3], batch[4]),
                        (batch[5], batch[6], batch[7], batch[8], batch[9]),
                    ]
                )

                for img_a, gt_a, img_b, gt_b, pose_gt in pairs:
                    pred_a, pred_b, _ = model(img_a, img_b)

                    Bsz, N, _ = gt_a.shape
                    dtype = gt_a.dtype

                    fov = torch.tensor(
                        np.deg2rad(fov_degrees),
                        device=device,
                        dtype=dtype,
                    )

                    theta = torch.linspace(
                        -0.5 * fov,
                        0.5 * fov,
                        N,
                        device=device,
                        dtype=dtype,
                    )

                    # Geometry source
                    depth_a = pred_a[..., 2] if use_pred_depth else gt_a[..., 2]
                    depth_b = pred_b[..., 2] if use_pred_depth else gt_b[..., 2]

                    radius_a = pred_a[..., 1] if use_pred_radius else gt_a[..., 1]
                    radius_b = pred_b[..., 1] if use_pred_radius else gt_b[..., 1]

                    occ_a = pred_a[..., 0] if use_pred_occ else gt_a[..., 0]
                    occ_b = pred_b[..., 0] if use_pred_occ else gt_b[..., 0]

                    pts_a = torch.stack([
                        depth_a * torch.cos(theta),
                        depth_a * torch.sin(theta),
                    ], dim=-1)

                    pts_b = torch.stack([
                        depth_b * torch.cos(theta),
                        depth_b * torch.sin(theta),
                    ], dim=-1)

                    for b in range(Bsz):
                        # GT cylinder IDs define correspondence.
                        # "GT coarse" means we are isolating geometry,
                        # not testing learned correspondence here.
                        ids_a = gt_a[b, :, 3].round().long()
                        ids_b = gt_b[b, :, 3].round().long()

                        if use_pred_occ:
                            mask_a = occ_a[b] > 0.05
                            mask_b = occ_b[b] > 0.05
                        else:
                            mask_a = occ_a[b] > occ_thresh
                            mask_b = occ_b[b] > occ_thresh

                        A, Bpts = [], []

                        for cid in torch.unique(ids_a[gt_a[b, :, 0] > occ_thresh]):
                            ia = torch.where(
                                mask_a & (ids_a == cid)
                            )[0]

                            ib = torch.where(
                                mask_b & (ids_b == cid)
                            )[0]

                            if ia.numel() != 1 or ib.numel() != 1:
                                continue

                            # Radius ablation:
                            # with predicted radius, only keep matches whose
                            # predicted radii are reasonably compatible.
                            if use_pred_radius:
                                ra = radius_a[b, ia[0]]
                                rb = radius_b[b, ib[0]]

                                rel_diff = torch.abs(ra - rb) / (
                                    torch.maximum(ra, rb) + 1e-6
                                )

                                if rel_diff > 0.5:
                                    continue

                            A.append(pts_a[b, ia[0]])
                            Bpts.append(pts_b[b, ib[0]])

                        matches_all.append(len(A))

                        if len(A) < 2:
                            zeros += 1
                            continue

                        A = torch.stack(A)
                        Bpts = torch.stack(Bpts)

                        result = ransac(A, Bpts)

                        if result is None:
                            zeros += 1
                            continue

                        R, t_ext, s_ext, c_ext, inliers = result
                        inliers_all.append(inliers.sum().item())

                        # ------------------------------------------
                        # Extrinsic A->B -> Camera2 pose in A
                        # ------------------------------------------
                        t_pose = -R.T @ t_ext
                        t_pose = F.normalize(t_pose, dim=0)

                        s_pose = -s_ext
                        c_pose = c_ext

                        # Translation direction error
                        t_gt = F.normalize(
                            pose_gt[b, :2],
                            dim=0,
                        )

                        cos_t = torch.clamp(
                            torch.dot(t_pose, t_gt),
                            -1.0,
                            1.0,
                        )

                        t_err = torch.rad2deg(
                            torch.acos(cos_t)
                        )

                        # Yaw error
                        yaw_gt = F.normalize(
                            pose_gt[b, 2:],
                            dim=0,
                        )

                        yaw_pred_angle = torch.atan2(
                            s_pose,
                            c_pose,
                        )

                        yaw_gt_angle = torch.atan2(
                            yaw_gt[0],
                            yaw_gt[1],
                        )

                        dyaw = torch.atan2(
                            torch.sin(yaw_pred_angle - yaw_gt_angle),
                            torch.cos(yaw_pred_angle - yaw_gt_angle),
                        )

                        yaw_err = torch.abs(
                            torch.rad2deg(dyaw)
                        )

                        t_errors.append(t_err.item())
                        yaw_errors.append(yaw_err.item())

        return {
            "t_mean": np.mean(t_errors) if t_errors else np.nan,
            "t_median": np.median(t_errors) if t_errors else np.nan,
            "t_p90": np.percentile(t_errors, 90) if t_errors else np.nan,
            "yaw_mean": np.mean(yaw_errors) if yaw_errors else np.nan,
            "yaw_median": np.median(yaw_errors) if yaw_errors else np.nan,
            "matches": np.mean(matches_all) if matches_all else 0,
            "inliers": np.mean(inliers_all) if inliers_all else 0,
            "zeros": zeros,
        }

    # --------------------------------------------------
    # Run all ablations
    # --------------------------------------------------
    for name, pd, pr, po in configs:
        results[name] = evaluate(pd, pr, po)

    print("RANSAC GEOMETRY ABLATION - GT COARSE — NEW POSE")
    print("-" * 112)

    print(
        f"{'Configuration':<24}"
        f"{'T mean':>9}"
        f"{'T med':>9}"
        f"{'T p90':>9}"
        f"{'Yaw mean':>10}"
        f"{'Yaw med':>9}"
        f"{'Matches':>10}"
        f"{'Inliers':>10}"
        f"{'Zeros':>8}"
    )

    for name, *_ in configs:
        r = results[name]

        print(
            f"{name:<24}"
            f"{r['t_mean']:9.2f}"
            f"{r['t_median']:9.2f}"
            f"{r['t_p90']:9.2f}"
            f"{r['yaw_mean']:10.2f}"
            f"{r['yaw_median']:9.2f}"
            f"{r['matches']:10.1f}"
            f"{r['inliers']:10.1f}"
            f"{r['zeros']:8d}"
        )

    return results

def test_gt_coarse_corr_ransac(
    model,
    loader,
    device,
    occ_thresh=0.5,
    fov_degrees=90.0,
):
    t_errors, yaw_errors = [], []

    with torch.no_grad():
        for batch in loader:
            batch = [x.to(device) for x in batch]

            pairs = [(batch[1], batch[3], batch[4])] if len(batch) == 5 else [
                (batch[1], batch[3], batch[4]),
                (batch[6], batch[8], batch[9]),
            ]

            for va, vb, pose_gt in pairs:
                B, N, _ = va.shape
                fov = torch.tensor(
                    np.deg2rad(fov_degrees),
                    device=device,
                    dtype=va.dtype,
                )

                theta = torch.linspace(
                    -0.5 * fov, 0.5 * fov, N,
                    device=device, dtype=va.dtype,
                )

                pts_a = torch.stack([
                    va[..., 2] * torch.cos(theta),
                    va[..., 2] * torch.sin(theta),
                ], -1)

                pts_b = torch.stack([
                    vb[..., 2] * torch.cos(theta),
                    vb[..., 2] * torch.sin(theta),
                ], -1)

                for b in range(B):
                    occ_a = va[b, :, 0] > occ_thresh
                    occ_b = vb[b, :, 0] > occ_thresh
                    ids_a = va[b, :, 3].round().long()
                    ids_b = vb[b, :, 3].round().long()

                    # Grupp: A image-column -> möjliga GT B-punkter
                    matches = []

                    for cid in torch.unique(ids_a[occ_a]):
                        ia = torch.where(occ_a & (ids_a == cid))[0]
                        ib = torch.where(occ_b & (ids_b == cid))[0]

                        if ia.numel() != 1 or ib.numel() != 1:
                            continue

                        theta_a = theta[ia[0]]
                        theta_b = theta[ib[0]]

                        xa = (
                            0.5
                            - 0.5 * torch.tan(theta_a)
                            / torch.tan(0.5 * fov)
                        )
                        xb = (
                            0.5
                            - 0.5 * torch.tan(theta_b)
                            / torch.tan(0.5 * fov)
                        )

                        ca = torch.clamp((xa * 16).long(), 0, 15)
                        cb = torch.clamp((xb * 16).long(), 0, 15)

                        matches.append((
                            ca.item(), cb.item(),
                            pts_a[b, ia[0]],
                            pts_b[b, ib[0]],
                        ))

                    # PoseHead har bara en match per A-column.
                    # Efterlikna det: medelvärde om flera cylindrar
                    # hamnar i samma coarse pair.
                    grouped = {}

                    for ca, cb, pa, pb in matches:
                        grouped.setdefault((ca, cb), [[], []])
                        grouped[(ca, cb)][0].append(pa)
                        grouped[(ca, cb)][1].append(pb)

                    if len(grouped) < 2:
                        continue

                    A, Bpts = [], []

                    for _, (aa, bb) in grouped.items():
                        A.append(torch.stack(aa).mean(0))
                        Bpts.append(torch.stack(bb).mean(0))

                    A = torch.stack(A)
                    Bpts = torch.stack(Bpts)

                    # Direct rigid alignment
                    ca = A.mean(0)
                    cb = Bpts.mean(0)
                    Ac = A - ca
                    Bc = Bpts - cb

                    dot = (
                        Ac[:, 0] * Bc[:, 0]
                        + Ac[:, 1] * Bc[:, 1]
                    ).sum()

                    cross = (
                        Ac[:, 0] * Bc[:, 1]
                        - Ac[:, 1] * Bc[:, 0]
                    ).sum()

                    norm = torch.sqrt(
                        dot.square() + cross.square() + 1e-8
                    )

                    c_ext = dot / norm
                    s_ext = cross / norm

                    R = torch.stack([
                        torch.stack([c_ext, -s_ext]),
                        torch.stack([s_ext,  c_ext]),
                    ])

                    t_ext = cb - R @ ca

                    # Extrinsic -> Camera2 pose in Camera1
                    t_pose = -R.T @ t_ext
                    t_pose = F.normalize(t_pose, dim=0)

                    yaw_pose = torch.stack([-s_ext, c_ext])

                    gt_t = F.normalize(pose_gt[b, :2], dim=0)
                    gt_yaw = F.normalize(pose_gt[b, 2:], dim=0)

                    t_cos = torch.clamp(
                        torch.dot(t_pose, gt_t), -1, 1
                    )
                    t_errors.append(
                        torch.rad2deg(torch.acos(t_cos)).item()
                    )

                    yp = torch.atan2(yaw_pose[0], yaw_pose[1])
                    yg = torch.atan2(gt_yaw[0], gt_yaw[1])

                    dy = torch.atan2(
                        torch.sin(yp - yg),
                        torch.cos(yp - yg),
                    )

                    yaw_errors.append(
                        torch.abs(torch.rad2deg(dy)).item()
                    )

    print("GT COARSE CORRESPONDENCE -> RIGID POSE")
    print("-" * 55)
    print(f"Samples:            {len(t_errors)}")
    print(f"Translation mean:   {np.mean(t_errors):.2f}°")
    print(f"Translation median: {np.median(t_errors):.2f}°")
    print(f"Yaw mean:           {np.mean(yaw_errors):.2f}°")
    print(f"Yaw median:         {np.median(yaw_errors):.2f}°")

    return t_errors, yaw_errors
