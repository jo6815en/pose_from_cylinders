import torch
import torch.nn.functional as F


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

    return total_vis, total_radius, total_pose, total_depth, total_t, total_r

def match_sinkhorn_between_views(
    vision_a,
    vision_b,
    pose_ab,
    occ_thresh=0.5,
    fov_degrees=90.0,
    lambda_pos=1.0,
    lambda_radius=1.0,
    temperature=0.1,
    sinkhorn_iters=50,
    dustbin_cost=1.0,
    min_match_prob=0.0,
):
    squeeze_batch = False
    if vision_a.dim() == 2:
        vision_a = vision_a.unsqueeze(0)
        vision_b = vision_b.unsqueeze(0)
        pose_ab = pose_ab.unsqueeze(0)
        squeeze_batch = True

    B, N, _ = vision_a.shape
    device = vision_a.device
    dtype = vision_a.dtype

    fov = torch.tensor(fov_degrees * torch.pi / 180.0, device=device, dtype=dtype)
    theta = torch.linspace(-0.5 * fov, 0.5 * fov, N, device=device, dtype=dtype)

    def vision_points_2d(vision):
        depth = vision[..., 2]
        x = depth * torch.cos(theta).view(1, N)
        y = depth * torch.sin(theta).view(1, N)
        return torch.stack([x, y], dim=-1)

    def sinkhorn(log_alpha, n_iters):
        for _ in range(n_iters):
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=0, keepdim=True)
        return log_alpha.exp()

    pts_a = vision_points_2d(vision_a)
    pts_b = vision_points_2d(vision_b)

    occ_a = vision_a[..., 0]
    rad_a = vision_a[..., 1]

    occ_b = vision_b[..., 0]
    rad_b = vision_b[..., 1]

    t = pose_ab[:, :2]

    yaw_vec = F.normalize(pose_ab[:, 2:], dim=-1)
    sin_yaw = yaw_vec[:, 0]
    cos_yaw = yaw_vec[:, 1]

    out = []

    for b in range(B):
        mask_a = occ_a[b] > occ_thresh
        mask_b = occ_b[b] > occ_thresh

        if mask_a.sum() == 0 or mask_b.sum() == 0:
            out.append({
                "indices_a": torch.empty(0, device=device, dtype=torch.long),
                "matched_indices_b": torch.empty(0, device=device, dtype=torch.long),
                "matched_cost": torch.empty(0, device=device, dtype=dtype),
                "match_prob": torch.empty(0, device=device, dtype=dtype),
                "dustbin_prob": torch.empty(0, device=device, dtype=dtype),
            })
            continue

        idx_a = torch.where(mask_a)[0]
        idx_b = torch.where(mask_b)[0]

        xa = pts_a[b, idx_a]
        xb = pts_b[b, idx_b]

        s = sin_yaw[b]
        c = cos_yaw[b]

        R = torch.stack([
            torch.stack([c, -s]),
            torch.stack([s,  c]),
        ])

        xa_in_b = xa @ R.T + t[b]

        pos_cost = torch.cdist(xa_in_b, xb, p=2)

        ra = rad_a[b, idx_a]
        rb = rad_b[b, idx_b]
        rad_cost = torch.abs(ra[:, None] - rb[None, :])

        cost = lambda_pos * pos_cost + lambda_radius * rad_cost

        n_a, n_b = cost.shape

        cost_ext = torch.full(
            (n_a + 1, n_b + 1),
            dustbin_cost,
            device=device,
            dtype=dtype,
        )

        cost_ext[:n_a, :n_b] = cost

        log_alpha = -cost_ext / temperature
        P_ext = sinkhorn(log_alpha, sinkhorn_iters)

        P_real = P_ext[:n_a, :n_b]
        P_dustbin_col = P_ext[:n_a, n_b]

        best_col_ext = P_ext[:n_a, :].argmax(dim=1)

        keep = best_col_ext < n_b

        if min_match_prob > 0.0:
            best_prob_ext = P_ext[
                torch.arange(n_a, device=device),
                best_col_ext
            ]
            keep = keep & (best_prob_ext >= min_match_prob)

        if keep.sum() == 0:
            out.append({
                "indices_a": torch.empty(0, device=device, dtype=torch.long),
                "matched_indices_b": torch.empty(0, device=device, dtype=torch.long),
                "matched_cost": torch.empty(0, device=device, dtype=dtype),
                "match_prob": torch.empty(0, device=device, dtype=dtype),
                "dustbin_prob": P_dustbin_col,
            })
            continue

        kept_rows = torch.where(keep)[0]
        best_local_b = best_col_ext[kept_rows]

        matched_idx_a = idx_a[kept_rows]
        matched_idx_b = idx_b[best_local_b]

        matched_cost = cost[kept_rows, best_local_b]
        match_prob = P_real[kept_rows, best_local_b]

        out.append({
            "indices_a": matched_idx_a,
            "matched_indices_b": matched_idx_b,
            "matched_cost": matched_cost,
            "match_prob": match_prob,
            "dustbin_prob": P_dustbin_col[kept_rows],
        })

    return out[0] if squeeze_batch else out


def match_vision_to_target_cylinders(
    pred_vision,
    target_vision,
    fov_degrees=90.0,
    occ_thresh=0.5,
    lambda_pos=1.0,
    lambda_radius=1.0,
):
    squeeze_batch = False
    if pred_vision.dim() == 2:
        pred_vision = pred_vision.unsqueeze(0)
        target_vision = target_vision.unsqueeze(0)
        squeeze_batch = True

    B, N, _ = pred_vision.shape
    device = pred_vision.device
    dtype = pred_vision.dtype

    fov = torch.tensor(fov_degrees * torch.pi / 180.0, device=device, dtype=dtype)
    theta = torch.linspace(-0.5 * fov, 0.5 * fov, N, device=device, dtype=dtype)

    pred_occ = pred_vision[..., 0]
    pred_rad = pred_vision[..., 1]
    pred_dep = pred_vision[..., 2]

    tgt_occ = target_vision[..., 0]
    tgt_rad = target_vision[..., 1]
    tgt_dep = target_vision[..., 2]
    tgt_id = target_vision[..., 3].long()

    x = torch.cos(theta).view(1, N)
    y = torch.sin(theta).view(1, N)

    pred_pts = torch.stack([pred_dep * x, pred_dep * y], dim=-1)
    tgt_pts = torch.stack([tgt_dep * x, tgt_dep * y], dim=-1)

    out = []

    for b in range(B):
        pred_mask = pred_occ[b] > occ_thresh
        tgt_mask = tgt_occ[b] > occ_thresh

        if pred_mask.sum() == 0 or tgt_mask.sum() == 0:
            out.append({
                "pred_indices": torch.empty(0, device=device, dtype=torch.long),
                "matched_ids": torch.empty(0, device=device, dtype=torch.long),
                "matched_target_indices": torch.empty(0, device=device, dtype=torch.long),
                "matched_cost": torch.empty(0, device=device, dtype=dtype),
            })
            continue

        pred_idx = torch.where(pred_mask)[0]
        tgt_idx = torch.where(tgt_mask)[0]

        p_pts = pred_pts[b, pred_idx]
        t_pts = tgt_pts[b, tgt_idx]

        p_rad = pred_rad[b, pred_idx]
        t_rad = tgt_rad[b, tgt_idx]

        pos_cost = torch.cdist(p_pts, t_pts, p=2)
        rad_cost = torch.abs(p_rad[:, None] - t_rad[None, :])

        cost = lambda_pos * pos_cost + lambda_radius * rad_cost

        best_local = cost.argmin(dim=1)
        best_target_idx = tgt_idx[best_local]

        out.append({
            "pred_indices": pred_idx,
            "matched_ids": tgt_id[b, best_target_idx],
            "matched_target_indices": best_target_idx,
            "matched_cost": cost[torch.arange(cost.shape[0], device=device), best_local],
        })

    return out[0] if squeeze_batch else out


def _get_pair_matches(
    pred_vision_a,
    target_vision_a=None,
    pred_vision_b=None,
    target_vision_b=None,
    relative_pose_pred=None,
    matching_mode="gt",
    occ_thresh=0.5,
    fov_degrees=90.0,
):
    if matching_mode == "gt":
        if target_vision_a is None or target_vision_b is None:
            raise ValueError("target_vision_a och target_vision_b krävs när matching_mode='gt'")

        match_a = match_vision_to_target_cylinders(
            pred_vision_a,
            target_vision_a,
            occ_thresh=occ_thresh,
            fov_degrees=fov_degrees,
        )

        match_b = match_vision_to_target_cylinders(
            pred_vision_b,
            target_vision_b,
            occ_thresh=occ_thresh,
            fov_degrees=fov_degrees,
        )

        return match_a, match_b

    if matching_mode == "sinkhorn":
        if relative_pose_pred is None:
            raise ValueError("relative_pose_pred krävs när matching_mode='sinkhorn'")

        match_ab = match_sinkhorn_between_views(
            pred_vision_a,
            pred_vision_b,
            relative_pose_pred,
            occ_thresh=occ_thresh,
            fov_degrees=fov_degrees,
        )

        return match_ab, None

    raise ValueError(f"Okänt matching_mode: {matching_mode}. Använd 'gt' eller 'sinkhorn'.")

def matched_radius_consistency_loss(
    pred_vision_a,
    target_vision_a=None,
    pred_vision_b=None,
    target_vision_b=None,
    relative_pose_pred=None,
    occ_thresh=0.5,
    fov_degrees=90.0,
    matching_mode="gt",
):
    match_a, match_b = _get_pair_matches(
        pred_vision_a,
        target_vision_a,
        pred_vision_b,
        target_vision_b,
        relative_pose_pred=relative_pose_pred,
        matching_mode=matching_mode,
        occ_thresh=occ_thresh,
        fov_degrees=fov_degrees,
    )

    if pred_vision_a.dim() == 2:
        pred_vision_a = pred_vision_a.unsqueeze(0)
        pred_vision_b = pred_vision_b.unsqueeze(0)
        match_a = [match_a]
        if match_b is not None:
            match_b = [match_b]

    losses = []

    if matching_mode == "sinkhorn":
        for batch_i, m in enumerate(match_a):
            idx_a = m["indices_a"]
            idx_b = m["matched_indices_b"]

            if idx_a.numel() == 0 or idx_b.numel() == 0:
                continue

            radius_a = pred_vision_a[batch_i, idx_a, 1]
            radius_b = pred_vision_b[batch_i, idx_b, 1]

            losses.append(F.l1_loss(radius_a, radius_b, reduction="mean"))

    else:
        for batch_i, (ma, mb) in enumerate(zip(match_a, match_b)):
            ids_a = ma["matched_ids"]
            ids_b = mb["matched_ids"]

            pred_idx_a = ma["pred_indices"]
            pred_idx_b = mb["pred_indices"]

            if ids_a.numel() == 0 or ids_b.numel() == 0:
                continue

            for cid in ids_a.unique():
                pos_a = torch.where(ids_a == cid)[0]
                pos_b = torch.where(ids_b == cid)[0]

                if pos_b.numel() == 0:
                    continue

                ia = pred_idx_a[pos_a[0]]
                ib = pred_idx_b[pos_b[0]]

                radius_a = pred_vision_a[batch_i, ia, 1]
                radius_b = pred_vision_b[batch_i, ib, 1]

                losses.append(F.l1_loss(radius_a, radius_b, reduction="mean"))

    if len(losses) == 0:
        return pred_vision_a.sum() * 0.0

    return torch.stack(losses).mean()

def matched_reprojection_loss_2d(
    pred_vision_a,
    target_vision_a=None,
    pred_vision_b=None,
    target_vision_b=None,
    relative_pose_pred=None,
    occ_thresh=0.5,
    fov_degrees=90.0,
    matching_mode="gt",
):
    if relative_pose_pred is None:
        raise ValueError("relative_pose_pred krävs för matched_reprojection_loss_2d")

    match_a, match_b = _get_pair_matches(
        pred_vision_a,
        target_vision_a,
        pred_vision_b,
        target_vision_b,
        relative_pose_pred=relative_pose_pred,
        matching_mode=matching_mode,
        occ_thresh=occ_thresh,
        fov_degrees=fov_degrees,
    )

    if pred_vision_a.dim() == 2:
        pred_vision_a = pred_vision_a.unsqueeze(0)
        pred_vision_b = pred_vision_b.unsqueeze(0)
        relative_pose_pred = relative_pose_pred.unsqueeze(0)
        match_a = [match_a]
        if match_b is not None:
            match_b = [match_b]

    B, N, _ = pred_vision_a.shape
    device = pred_vision_a.device
    dtype = pred_vision_a.dtype

    fov = torch.tensor(fov_degrees * torch.pi / 180.0, device=device, dtype=dtype)
    theta = torch.linspace(-0.5 * fov, 0.5 * fov, N, device=device, dtype=dtype)

    def vision_points_2d(vision):
        depth = vision[..., 2]
        x = depth * torch.cos(theta).view(1, N)
        y = depth * torch.sin(theta).view(1, N)
        return torch.stack([x, y], dim=-1)

    pts_a = vision_points_2d(pred_vision_a)
    pts_b = vision_points_2d(pred_vision_b)

    t = relative_pose_pred[:, :2]

    yaw_vec = F.normalize(relative_pose_pred[:, 2:], dim=-1)
    sin_yaw = yaw_vec[:, 0]
    cos_yaw = yaw_vec[:, 1]

    losses = []

    for batch_i in range(B):
        if matching_mode == "sinkhorn":
            m = match_a[batch_i]

            idx_a = m["indices_a"]
            idx_b = m["matched_indices_b"]

            if idx_a.numel() == 0 or idx_b.numel() == 0:
                continue

            pairs = zip(idx_a, idx_b)

        else:
            ma = match_a[batch_i]
            mb = match_b[batch_i]

            ids_a = ma["matched_ids"]
            ids_b = mb["matched_ids"]

            pred_idx_a = ma["pred_indices"]
            pred_idx_b = mb["pred_indices"]

            if ids_a.numel() == 0 or ids_b.numel() == 0:
                continue

            pairs = []

            for cid in ids_a.unique():
                pos_a = torch.where(ids_a == cid)[0]
                pos_b = torch.where(ids_b == cid)[0]

                if pos_b.numel() == 0:
                    continue

                pairs.append((pred_idx_a[pos_a[0]], pred_idx_b[pos_b[0]]))

        s = sin_yaw[batch_i]
        c = cos_yaw[batch_i]

        R = torch.stack([
            torch.stack([c, -s]),
            torch.stack([s,  c]),
        ])

        for ia, ib in pairs:
            x_a = pts_a[batch_i, ia]
            x_b = pts_b[batch_i, ib]

            x_b_pred = R @ x_a + t[batch_i]

            losses.append(F.smooth_l1_loss(x_b_pred, x_b, reduction="mean"))

    if len(losses) == 0:
        return pred_vision_a.sum() * 0.0
    

    return torch.stack(losses).mean()

def vision_loss(
    pred_vision,
    target_vision,
    occ_thresh=0.5,
    lambda_occ=1.0,
    lambda_radius=1.0,
    lambda_depth=1.0,
):
    pred_occ = pred_vision[..., 0]
    pred_rad = pred_vision[..., 1]
    pred_dep = pred_vision[..., 2]

    tgt_occ = target_vision[..., 0]
    tgt_rad = target_vision[..., 1]
    tgt_dep = target_vision[..., 2]

    occ_loss = F.binary_cross_entropy(
        pred_occ.clamp(1e-6, 1.0 - 1e-6),
        tgt_occ
    )

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
    vis_a_total, vis_a_occ, vis_a_rad, vis_a_dep = vision_loss(
        pred_vision_a,
        target_vision_a,
        occ_thresh=occ_thresh,
        lambda_occ=lambda_occ,
        lambda_radius=lambda_radius,
        lambda_depth=lambda_depth,
    )

    vis_b_total, vis_b_occ, vis_b_rad, vis_b_dep = vision_loss(
        pred_vision_b,
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

    total = (
        lambda_vis * vis_a_total +
        lambda_vis * vis_b_total
    ) / 2.0 + lambda_pose * pose_total

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

        f1 = torch.stack([rad1[m1], dep1[m1]], dim=-1)
        f2 = torch.stack([rad2[m2], dep2[m2]], dim=-1)

        o1 = occ1[m1]
        o2 = occ2[m2]

        cost = torch.cdist(f1, f2, p=1)

        nn12 = cost.argmin(dim=1)
        nn21 = cost.argmin(dim=0)

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

        loss_occ = F.mse_loss(o1[idx1], o2[idx2])
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