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
                pred_a, pred_b, _ = model(img_a, img_b,)

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
                    pred_a, pred_b, _ = model(img_a, img_b, )

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

def full_pred_ransac(model, val_loader, device):

    model.eval()

    t_errors = []
    yaw_errors = []
    zeros = 0

    with torch.no_grad():
        for batch in val_loader:
            img_a, vision_a, img_b, vision_b, pose_gt = (
                x.to(device, non_blocking=True) for x in batch
            )

            _, _, pose_pred = model.predict_pose_ransac(img_a, img_b)

            for b in range(pose_pred.shape[0]):

                # RANSAC kunde inte estimera pose
                if torch.linalg.vector_norm(pose_pred[b]) < 1e-6:
                    zeros += 1
                    continue

                t_pred = F.normalize(pose_pred[b, :2], dim=0)

                t_gt = F.normalize(pose_gt[b, :2], dim=0)

                cos_t = torch.clamp(
                    torch.dot(t_pred, t_gt),
                    -1.0,
                    1.0,
                )

                t_err = torch.rad2deg(torch.acos(cos_t))

                yaw_pred = F.normalize(pose_pred[b, 2:], dim=0)

                yaw_gt = F.normalize(pose_gt[b, 2:], dim=0)

                a_pred = torch.atan2(yaw_pred[0], yaw_pred[1]) 

                a_gt = torch.atan2(yaw_gt[0], yaw_gt[1])

                dyaw = torch.atan2(
                    torch.sin(a_pred - a_gt),
                    torch.cos(a_pred - a_gt),
                )

                yaw_err = torch.abs(torch.rad2deg(dyaw))

                t_errors.append(t_err.item())
                yaw_errors.append(yaw_err.item())


    t_errors = np.asarray(t_errors)
    yaw_errors = np.asarray(yaw_errors)

    print("FULL PREDICTED RANSAC")
    print("-" * 50)
    print(f"Samples evaluated:    {len(t_errors)}")
    print(f"Zeros:                {zeros}")
    print()
    print(f"Translation mean:     {t_errors.mean():.2f}°")
    print(f"Translation median:   {np.median(t_errors):.2f}°")
    print(f"Translation p90:      {np.percentile(t_errors, 90):.2f}°")
    print()
    print(f"Yaw mean:             {yaw_errors.mean():.2f}°")
    print(f"Yaw median:           {np.median(yaw_errors):.2f}°")
    print(f"Yaw p90:              {np.percentile(yaw_errors, 90):.2f}°")


def test_gt_pose_predicted_geometry(
    model, loader, device,
    occ_thresh=0.5, fov_degrees=90.0,
):
    model.eval()

    gt_residuals = []
    pred_residuals = []
    pred_a_to_gt_b = []
    gt_a_to_pred_b = []

    with torch.no_grad():
        for batch in loader:
            batch = [x.to(device, non_blocking=True) for x in batch]

            pairs = (
                [(batch[0], batch[1], batch[2], batch[3], batch[4])]
                if len(batch) == 5 else
                [
                    (batch[0], batch[1], batch[2], batch[3], batch[4]),
                    (batch[5], batch[6], batch[7], batch[8], batch[9]),
                ]
            )

            for img_a, gt_a, img_b, gt_b, pose_gt in pairs:
                pred_a, pred_b, _ = model(
                    img_a, img_b,
                )

                B, N, _ = gt_a.shape
                fov = torch.tensor(
                    np.deg2rad(fov_degrees),
                    device=device, dtype=gt_a.dtype,
                )
                theta = torch.linspace(
                    -0.5*fov, 0.5*fov, N,
                    device=device, dtype=gt_a.dtype,
                )

                gt_pts_a = torch.stack([
                    gt_a[..., 2] * torch.cos(theta),
                    gt_a[..., 2] * torch.sin(theta),
                ], dim=-1)

                gt_pts_b = torch.stack([
                    gt_b[..., 2] * torch.cos(theta),
                    gt_b[..., 2] * torch.sin(theta),
                ], dim=-1)

                pred_pts_a = torch.stack([
                    pred_a[..., 2] * torch.cos(theta),
                    pred_a[..., 2] * torch.sin(theta),
                ], dim=-1)

                pred_pts_b = torch.stack([
                    pred_b[..., 2] * torch.cos(theta),
                    pred_b[..., 2] * torch.sin(theta),
                ], dim=-1)

                for b in range(B):
                    occ_a = gt_a[b, :, 0] > occ_thresh
                    occ_b = gt_b[b, :, 0] > occ_thresh
                    ids_a = gt_a[b, :, 3].round().long()
                    ids_b = gt_b[b, :, 3].round().long()

                    t = pose_gt[b, :2]
                    yaw = F.normalize(pose_gt[b, 2:], dim=0)
                    s, c = yaw[0], yaw[1]

                    # Samma transform som i vårt verifierade GT geometry-test:
                    # punkt i A-frame -> B-frame
                    R_inv = torch.stack([
                        torch.stack([ c,  s]),
                        torch.stack([-s,  c]),
                    ])

                    for cid in torch.unique(ids_a[occ_a]):
                        ia = torch.where(occ_a & (ids_a == cid))[0]
                        ib = torch.where(occ_b & (ids_b == cid))[0]

                        if ia.numel() != 1 or ib.numel() != 1:
                            continue

                        A_gt = gt_pts_a[b, ia[0]]
                        B_gt = gt_pts_b[b, ib[0]]
                        A_pred = pred_pts_a[b, ia[0]]
                        B_pred = pred_pts_b[b, ib[0]]

                        A_gt_in_B = R_inv @ (A_gt - t)
                        A_pred_in_B = R_inv @ (A_pred - t)

                        # Sanity floor: GT geometry + GT pose
                        gt_residuals.append(
                            torch.linalg.vector_norm(
                                A_gt_in_B - B_gt
                            ).item()
                        )

                        # Huvudmåttet:
                        # predicted geometry A -> GT pose -> predicted geometry B
                        pred_residuals.append(
                            torch.linalg.vector_norm(
                                A_pred_in_B - B_pred
                            ).item()
                        )

                        # Diagnostik: vilken sida bidrar med fel?
                        pred_a_to_gt_b.append(
                            torch.linalg.vector_norm(
                                A_pred_in_B - B_gt
                            ).item()
                        )

                        gt_a_to_pred_b.append(
                            torch.linalg.vector_norm(
                                A_gt_in_B - B_pred
                            ).item()
                        )

    gt_residuals = np.asarray(gt_residuals)
    pred_residuals = np.asarray(pred_residuals)
    pred_a_to_gt_b = np.asarray(pred_a_to_gt_b)
    gt_a_to_pred_b = np.asarray(gt_a_to_pred_b)

    def stats(name, x):
        print(
            f"{name:<30}"
            f" mean={x.mean():6.3f} m"
            f"  median={np.median(x):6.3f} m"
            f"  p90={np.percentile(x, 90):6.3f} m"
        )

    print("GT POSE -> PREDICTED GEOMETRY CONSISTENCY")
    print("-" * 80)
    print(f"Matches: {len(pred_residuals)}")
    print()

    stats("GT A -> GT B", gt_residuals)
    stats("Pred A -> GT B", pred_a_to_gt_b)
    stats("GT A -> Pred B", gt_a_to_pred_b)
    stats("Pred A -> Pred B", pred_residuals)

    return {
        "gt_residual": gt_residuals,
        "pred_a_to_gt_b": pred_a_to_gt_b,
        "gt_a_to_pred_b": gt_a_to_pred_b,
        "pred_residual": pred_residuals,
    }

def test_depth_by_apparent_width(
    model, loader, device,
    occ_thresh=0.5,
):
    model.eval()

    widths_all, pred_all, gt_all = [], [], []

    with torch.no_grad():
        for batch in loader:
            batch = [x.to(device, non_blocking=True) for x in batch]

            pairs = (
                [(batch[0], batch[1]), (batch[2], batch[3])]
                if len(batch) == 5 else
                [(batch[0], batch[1]), (batch[2], batch[3]),
                 (batch[5], batch[6]), (batch[7], batch[8])]
            )

            for img, gt in pairs:
                pred, _, _ = model(img, img,)

                mask = gt[..., 0] > occ_thresh

                gt_depth = gt[..., 2][mask]
                gt_radius = gt[..., 1][mask]
                pred_depth = pred[..., 2][mask]

                # Samma geometri som project_cylinder:
                # full angular width = 2 * asin(radius / radial_depth)
                width_rad = 2.0 * torch.asin(
                    torch.clamp(gt_radius / gt_depth, 0.0, 1.0)
                )
                width_deg = torch.rad2deg(width_rad)

                widths_all.append(width_deg.cpu())
                pred_all.append(pred_depth.cpu())
                gt_all.append(gt_depth.cpu())

    widths = torch.cat(widths_all).numpy()
    pred = torch.cat(pred_all).numpy()
    gt = torch.cat(gt_all).numpy()

    bins = [
        (0, 2),
        (2, 4),
        (4, 8),
        (8, 16),
        (16, np.inf),
    ]

    def corr(a, b):
        if len(a) < 2 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
            return np.nan
        return np.corrcoef(a, b)[0, 1]

    print("DEPTH ERROR BY APPARENT CYLINDER WIDTH")
    print("-" * 100)
    print(
        f"{'Width':>10} {'N':>7} {'MAE':>8} {'Bias':>8} {'Corr':>8} "
        f"{'Pred mean':>10} {'GT mean':>9} {'Pred std':>10} {'GT std':>9}"
    )

    results = {}

    for lo, hi in bins:
        m = (widths >= lo) & (widths < hi)
        if not m.any():
            continue

        p, g = pred[m], gt[m]
        label = f"{lo}-{hi:g}°" if np.isfinite(hi) else f"{lo}+°"

        mae = np.mean(np.abs(p - g))
        bias = np.mean(p - g)
        c = corr(p, g)

        print(
            f"{label:>10} {len(g):7d} {mae:8.3f} {bias:8.3f} {c:8.3f} "
            f"{p.mean():10.3f} {g.mean():9.3f} {p.std():10.3f} {g.std():9.3f}"
        )

        results[label] = {
            "n": len(g),
            "mae": mae,
            "bias": bias,
            "corr": c,
            "pred_mean": p.mean(),
            "gt_mean": g.mean(),
            "pred_std": p.std(),
            "gt_std": g.std(),
        }

    print("\nGLOBAL RELATIONSHIPS")
    print("-" * 55)
    print(f"Width ↔ GT depth:          {corr(widths, gt):.3f}")
    print(f"Width ↔ predicted depth:   {corr(widths, pred):.3f}")
    print(f"Pred depth ↔ GT depth:     {corr(pred, gt):.3f}")

    return {
        "width_deg": widths,
        "pred_depth": pred,
        "gt_depth": gt,
        "bins": results,
    }

def test_radius_size_depth_cue(model, loader, device, occ_thresh=0.5):
    model.eval()

    gt_depths, pred_depths = [], []
    depth_gt_radius, depth_pred_radius = [], []
    gt_radii, pred_radii, widths = [], [], []

    with torch.no_grad():
        for batch in loader:
            batch = [x.to(device, non_blocking=True) for x in batch]

            pairs = (
                [(batch[0], batch[1]), (batch[2], batch[3])]
                if len(batch) == 5 else
                [(batch[0], batch[1]), (batch[2], batch[3]),
                 (batch[5], batch[6]), (batch[7], batch[8])]
            )

            for img, gt in pairs:
                pred, _, _ = model(img, img,)
                mask = gt[..., 0] > occ_thresh

                gd = gt[..., 2][mask]
                gr = gt[..., 1][mask]
                pd = pred[..., 2][mask]
                pr = pred[..., 1][mask]

                # Exact apparent angular half-width from GT geometry
                alpha = torch.asin(torch.clamp(gr / gd, 0.0, 1.0))
                sin_alpha = torch.sin(alpha).clamp_min(1e-6)

                # Sanity: GT radius + exact apparent width
                d_from_gt_radius = gr / sin_alpha

                # Key test: predicted radius + exact apparent width
                d_from_pred_radius = pr / sin_alpha

                gt_depths.append(gd.cpu())
                pred_depths.append(pd.cpu())
                depth_gt_radius.append(d_from_gt_radius.cpu())
                depth_pred_radius.append(d_from_pred_radius.cpu())
                gt_radii.append(gr.cpu())
                pred_radii.append(pr.cpu())
                widths.append(torch.rad2deg(2.0 * alpha).cpu())

    gd = torch.cat(gt_depths).numpy()
    pd = torch.cat(pred_depths).numpy()
    dgr = torch.cat(depth_gt_radius).numpy()
    dpr = torch.cat(depth_pred_radius).numpy()
    gr = torch.cat(gt_radii).numpy()
    pr = torch.cat(pred_radii).numpy()
    w = torch.cat(widths).numpy()

    def corr(a, b):
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            return np.nan
        return np.corrcoef(a, b)[0, 1]

    def show(name, p):
        print(
            f"{name:<30}"
            f" MAE={np.mean(np.abs(p-gd)):6.3f} m"
            f"  corr={corr(p, gd):6.3f}"
            f"  std={p.std():6.3f}"
        )

    print("RADIUS + APPARENT SIZE DEPTH TEST")
    print("-" * 75)
    print(f"Cylinders: {len(gd)}")
    print()

    show("Network predicted depth", pd)
    show("GT radius + width", dgr)
    show("Pred radius + width", dpr)

    print()
    print(f"GT radius ↔ predicted radius: {corr(gr, pr):.3f}")
    print(f"GT radius MAE:                {np.mean(np.abs(pr-gr)):.3f} m")
    print(f"Width ↔ GT depth:             {corr(w, gd):.3f}")

    return {
        "gt_depth": gd,
        "pred_depth": pd,
        "depth_from_gt_radius": dgr,
        "depth_from_pred_radius": dpr,
        "gt_radius": gr,
        "pred_radius": pr,
        "width_deg": w,
    }

def test_ransac_depth_corr_ablation(
    model, loader, device,
    occ_thresh=0.5, temperature=0.1,
    search_radius=16, inlier_threshold=0.4,
    num_iters=200, fov_degrees=90.0,
):
    model.eval()

    configs = [
        ("GT depth + GT corr", False, False),
        ("Pred depth + GT corr", True, False),
        ("GT depth + Pred corr", False, True),
        ("Pred depth + Pred corr", True, True),
    ]

    results = {
        name: {"t": [], "yaw": [], "matches": [], "inliers": [], "zeros": 0}
        for name, _, _ in configs
    }

    def rigid(A, B):
        ca, cb = A.mean(0), B.mean(0)
        Ac, Bc = A - ca, B - cb
        dot = (Ac[:, 0]*Bc[:, 0] + Ac[:, 1]*Bc[:, 1]).sum()
        cross = (Ac[:, 0]*Bc[:, 1] - Ac[:, 1]*Bc[:, 0]).sum()
        norm = torch.sqrt(dot.square() + cross.square() + 1e-8)
        c, s = dot/norm, cross/norm
        R = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])
        return R, cb - R @ ca, s, c

    def ransac(A, B):
        if len(A) < 2:
            return None

        best, best_count, best_error = None, 0, float("inf")

        for _ in range(num_iters):
            idx = torch.randperm(len(A), device=A.device)[:2]
            R, t, _, _ = rigid(A[idx], B[idx])
            res = torch.linalg.vector_norm(A @ R.T + t - B, dim=-1)
            mask = res < inlier_threshold
            count = mask.sum().item()

            if count < 2:
                continue

            error = res[mask].mean().item()
            if count > best_count or (count == best_count and error < best_error):
                best, best_count, best_error = mask, count, error

        if best is None:
            return None

        R, t, s, c = rigid(A[best], B[best])
        return R, t, s, c, best

    with torch.no_grad():
        for batch in loader:
            batch = [x.to(device, non_blocking=True) for x in batch]

            pairs = (
                [(batch[0], batch[1], batch[2], batch[3], batch[4])]
                if len(batch) == 5 else
                [
                    (batch[0], batch[1], batch[2], batch[3], batch[4]),
                    (batch[5], batch[6], batch[7], batch[8], batch[9]),
                ]
            )

            for img_a, gt_a, img_b, gt_b, pose_gt in pairs:
                pred_a, pred_b, _, corr_a, corr_b = model(
                    img_a, img_b,  return_corr=True
                )

                corr_a = F.normalize(corr_a, dim=-1)
                corr_b = F.normalize(corr_b, dim=-1)
                prob = F.softmax(
                    torch.matmul(corr_a, corr_b.transpose(1, 2)) / temperature,
                    dim=-1,
                )

                Bsz, N, _ = gt_a.shape
                fov = torch.tensor(
                    np.deg2rad(fov_degrees), device=device, dtype=gt_a.dtype
                )
                theta = torch.linspace(
                    -0.5*fov, 0.5*fov, N, device=device, dtype=gt_a.dtype
                )

                for name, use_pred_depth, use_pred_corr in configs:
                    da = pred_a[..., 2] if use_pred_depth else gt_a[..., 2]
                    db = pred_b[..., 2] if use_pred_depth else gt_b[..., 2]

                    pts_a = torch.stack(
                        [da*torch.cos(theta), da*torch.sin(theta)], dim=-1
                    )
                    pts_b = torch.stack(
                        [db*torch.cos(theta), db*torch.sin(theta)], dim=-1
                    )

                    for b in range(Bsz):
                        occ_a = gt_a[b, :, 0] > occ_thresh
                        occ_b = gt_b[b, :, 0] > occ_thresh
                        ids_a = gt_a[b, :, 3].round().long()
                        ids_b = gt_b[b, :, 3].round().long()

                        A, Bpts = [], []

                        for cid in torch.unique(ids_a[occ_a]):
                            ia = torch.where(occ_a & (ids_a == cid))[0]
                            ib_gt = torch.where(occ_b & (ids_b == cid))[0]

                            if ia.numel() != 1 or ib_gt.numel() != 1:
                                continue

                            ia = ia[0]

                            if not use_pred_corr:
                                ib = ib_gt[0]

                            else:
                                # A geometry-bin -> 16-column image coordinate
                                theta_a = theta[ia]
                                x_a = 0.5 - 0.5 * torch.tan(theta_a) / torch.tan(0.5*fov)
                                col_a = torch.clamp((x_a * 16).long(), 0, 15)

                                # Predicted coarse B column
                                col_b = prob[b, col_a].argmax()

                                # B column -> geometry-bin neighbourhood
                                x_b = (col_b.to(gt_a.dtype) + 0.5) / 16.0
                                theta_b = torch.atan(
                                    (1.0 - 2.0*x_b) * torch.tan(0.5*fov)
                                )
                                center = torch.round(
                                    (theta_b + 0.5*fov) / fov * (N - 1)
                                ).long().clamp(0, N - 1)

                                lo = max(0, center.item() - search_radius)
                                hi = min(N, center.item() + search_radius + 1)
                                candidates = torch.arange(lo, hi, device=device)
                                candidates = candidates[occ_b[candidates]]

                                if candidates.numel() == 0:
                                    continue

                                # Keep radius GT here: isolate coarse correspondence.
                                radius_a = gt_a[b, ia, 1]
                                radius_diff = torch.abs(
                                    gt_b[b, candidates, 1] - radius_a
                                )
                                ib = candidates[radius_diff.argmin()]

                            A.append(pts_a[b, ia])
                            Bpts.append(pts_b[b, ib])

                        results[name]["matches"].append(len(A))

                        if len(A) < 2:
                            results[name]["zeros"] += 1
                            continue

                        result = ransac(torch.stack(A), torch.stack(Bpts))

                        if result is None:
                            results[name]["zeros"] += 1
                            continue

                        R, t_ext, s_ext, c_ext, inliers = result
                        t_pose = F.normalize(-R.T @ t_ext, dim=0)
                        t_gt = F.normalize(pose_gt[b, :2], dim=0)

                        cos_t = torch.clamp(torch.dot(t_pose, t_gt), -1.0, 1.0)
                        t_err = torch.rad2deg(torch.acos(cos_t)).item()

                        yaw_gt = F.normalize(pose_gt[b, 2:], dim=0)
                        pred_angle = torch.atan2(-s_ext, c_ext)
                        gt_angle = torch.atan2(yaw_gt[0], yaw_gt[1])
                        dyaw = torch.atan2(
                            torch.sin(pred_angle - gt_angle),
                            torch.cos(pred_angle - gt_angle),
                        )

                        results[name]["t"].append(t_err)
                        results[name]["yaw"].append(
                            torch.abs(torch.rad2deg(dyaw)).item()
                        )
                        results[name]["inliers"].append(inliers.sum().item())

    print("RANSAC DEPTH x COARSE CORRESPONDENCE ABLATION")
    print("-" * 108)
    print(
        f"{'Configuration':<28}"
        f"{'T mean':>10}{'T med':>10}{'T p90':>10}"
        f"{'Yaw mean':>11}{'Yaw med':>10}"
        f"{'Matches':>10}{'Inliers':>10}{'Zeros':>8}"
    )

    summary = {}

    for name, _, _ in configs:
        r = results[name]
        t = np.asarray(r["t"])
        yaw = np.asarray(r["yaw"])

        summary[name] = {
            "t_mean": t.mean(),
            "t_median": np.median(t),
            "t_p90": np.percentile(t, 90),
            "yaw_mean": yaw.mean(),
            "yaw_median": np.median(yaw),
            "matches": np.mean(r["matches"]),
            "inliers": np.mean(r["inliers"]),
            "zeros": r["zeros"],
        }

        s = summary[name]
        print(
            f"{name:<28}"
            f"{s['t_mean']:10.2f}{s['t_median']:10.2f}{s['t_p90']:10.2f}"
            f"{s['yaw_mean']:11.2f}{s['yaw_median']:10.2f}"
            f"{s['matches']:10.1f}{s['inliers']:10.1f}{s['zeros']:8d}"
        )

    return summary

def evaluate_width_prediction(model, loader, device, occ_thresh=0.5):
    model.eval()
    pred_all, gt_all = [], []

    with torch.no_grad():
        for batch in loader:
            batch = [x.to(device, non_blocking=True) for x in batch]

            pairs = (
                [(batch[0], batch[1]), (batch[2], batch[3])]
                if len(batch) == 5 else
                [(batch[0], batch[1]), (batch[2], batch[3]),
                 (batch[5], batch[6]), (batch[7], batch[8])]
            )

            for img, gt in pairs:
                pred, _, _ = model(img, img,)
                mask = gt[..., 0] > occ_thresh

                gt_width = torch.rad2deg(
                    2.0 * torch.asin(
                        torch.clamp(
                            gt[..., 1] / gt[..., 2].clamp_min(1e-6),
                            0.0, 0.9999,
                        )
                    )
                )

                pred_all.append(pred[..., 3][mask].cpu())
                gt_all.append(gt_width[mask].cpu())

    pred = torch.cat(pred_all).numpy()
    gt = torch.cat(gt_all).numpy()

    print("APPARENT WIDTH PREDICTION")
    print("-" * 50)
    print(f"MAE:              {np.mean(np.abs(pred - gt)):.3f}°")
    print(f"Correlation:      {np.corrcoef(pred, gt)[0,1]:.3f}")
    print(f"Pred mean:        {pred.mean():.3f}°")
    print(f"GT mean:          {gt.mean():.3f}°")
    print(f"Pred std:         {pred.std():.3f}°")
    print(f"GT std:           {gt.std():.3f}°")

    return {"pred": pred, "gt": gt}