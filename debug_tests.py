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

def test_corr_descriptor_diversity(model, loader, device):
    model.eval()

    adjacent_sims = []
    random_sims = []
    descriptor_stds = []

    with torch.no_grad():
        for batch in loader:
            img_a, vision_a, img_b, vision_b, _ = (x.to(device, non_blocking=True) for x in batch)

            _, _, _, corr_a, corr_b = model(img_a, img_b, return_corr=True)

            for corr in [corr_a, corr_b]:
                corr = F.normalize(corr, dim=-1)

                # Likhet mellan intilliggande bins
                adjacent = F.cosine_similarity(corr[:, :-1], corr[:, 1:], dim=-1)
                adjacent_sims.extend(adjacent.flatten().cpu().tolist())

                # Likhet mellan bins långt ifrån varandra
                random = F.cosine_similarity(corr[:, :64], corr[:, 64:], dim=-1)
                random_sims.extend(random.flatten().cpu().tolist())

                # Hur mycket descriptorerna varierar inom varje sample
                descriptor_stds.extend(corr.std(dim=1).mean(dim=-1).cpu().tolist())

    print("CORRESPONDENCE DESCRIPTOR DIVERSITY")
    print("----------------------------------------")
    print(f"Adjacent-bin cosine similarity: {np.mean(adjacent_sims):.4f}")
    print(f"Distant-bin cosine similarity:  {np.mean(random_sims):.4f}")
    print(f"Descriptor std across bins:     {np.mean(descriptor_stds):.4f}")

def test_correspondence_position_bias(model, loader, device, occ_thresh=0.5, temperature=0.1):
    model.eval()

    pred_shifts = []
    gt_shifts = []
    shift_errors = []
    same_bin = 0
    near_same_bin = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            batch = [x.to(device, non_blocking=True) for x in batch]

            if len(batch) == 5:
                pairs = [(batch[0], batch[1], batch[2], batch[3])]
            elif len(batch) == 10:
                pairs = [
                    (batch[0], batch[1], batch[2], batch[3]),
                    (batch[5], batch[6], batch[7], batch[8]),
                ]
            else:
                raise ValueError(f"Unexpected batch length: {len(batch)}")

            for img_a, vision_a, img_b, vision_b in pairs:
                _, _, _, corr_a, corr_b = model(img_a, img_b, return_corr=True)

                corr_a = F.normalize(corr_a, dim=-1)
                corr_b = F.normalize(corr_b, dim=-1)

                for b in range(img_a.shape[0]):
                    idx_a = torch.where(vision_a[b, :, 0] > occ_thresh)[0]
                    idx_b = torch.where(vision_b[b, :, 0] > occ_thresh)[0]

                    ids_a = vision_a[b, idx_a, 3].round().long()
                    ids_b = vision_b[b, idx_b, 3].round().long()

                    sim = corr_a[b] @ corr_b[b].T / temperature
                    pred_b = sim.argmax(dim=-1)

                    for cid in ids_a.unique():
                        match_a = idx_a[ids_a == cid]
                        match_b = idx_b[ids_b == cid]

                        if match_a.numel() != 1 or match_b.numel() != 1:
                            continue

                        ia = match_a[0].item()
                        ib_gt = match_b[0].item()
                        ib_pred = pred_b[ia].item()

                        pred_shift = ib_pred - ia
                        gt_shift = ib_gt - ia

                        pred_shifts.append(pred_shift)
                        gt_shifts.append(gt_shift)
                        shift_errors.append(abs(pred_shift - gt_shift))

                        same_bin += abs(pred_shift) == 0
                        near_same_bin += abs(pred_shift) <= 2
                        total += 1

    pred_shifts = np.array(pred_shifts)
    gt_shifts = np.array(gt_shifts)
    shift_errors = np.array(shift_errors)

    print("CORRESPONDENCE POSITION BIAS")
    print("----------------------------------------")
    print(f"Matches:                     {total}")
    print(f"Predicted same bin:          {100 * same_bin / total:.1f}%")
    print(f"Predicted within ±2 of A:    {100 * near_same_bin / total:.1f}%")
    print(f"Mean |predicted shift|:      {np.mean(np.abs(pred_shifts)):.2f} bins")
    print(f"Mean |GT shift|:             {np.mean(np.abs(gt_shifts)):.2f} bins")
    print(f"Mean shift error:            {np.mean(shift_errors):.2f} bins")
    print(f"Correlation pred vs GT shift:{np.corrcoef(pred_shifts, gt_shifts)[0,1]:.3f}")

    return pred_shifts, gt_shifts


def geometric_translation_error(target_a, target_b, pose_gt, pose_pred, fov_degrees=90.0):
    B, N, _ = target_a.shape
    device, dtype = target_a.device, target_a.dtype

    fov = torch.tensor(float(fov_degrees) * torch.pi / 180.0, device=device, dtype=dtype)
    theta = torch.linspace(-0.5 * fov, 0.5 * fov, N, device=device, dtype=dtype)

    def points(vision):
        depth = vision[..., 2]
        return torch.stack([depth * torch.cos(theta), depth * torch.sin(theta)], dim=-1)

    pts_a, pts_b = points(target_a), points(target_b)
    errors, pred_directions, gt_directions = [], [], []

    for b in range(B):
        occ_a = target_a[b, :, 0] > 0.5
        occ_b = target_b[b, :, 0] > 0.5
        ids_a = target_a[b, :, 3].round().long()
        ids_b = target_b[b, :, 3].round().long()

        yaw = F.normalize(pose_pred[b, 2:], dim=0)
        s, c = yaw[0], yaw[1]
        R = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])

        translations = []
        for cid in torch.unique(ids_a[occ_a]):
            ia = torch.where(occ_a & (ids_a == cid))[0]
            ib = torch.where(occ_b & (ids_b == cid))[0]

            if ia.numel() != 1 or ib.numel() != 1:
                continue

            pa, pb = pts_a[b, ia[0]], pts_b[b, ib[0]]
            translations.append(pb - R @ pa)

        if not translations:
            continue

        t_pred = F.normalize(torch.stack(translations).median(dim=0).values, dim=0)
        t_gt = F.normalize(pose_gt[b, :2], dim=0)

        cos_sim = torch.clamp(torch.dot(t_pred, t_gt), -1.0, 1.0)
        angle = torch.rad2deg(torch.acos(cos_sim))

        errors.append(angle.item())
        pred_directions.append(t_pred.cpu())
        gt_directions.append(t_gt.cpu())

    return errors, pred_directions, gt_directions


def test_1(model, val_loader, device):
    model.eval()
    all_errors = []

    with torch.no_grad():
        for batch in val_loader:
            img_a, vision_a, img_b, vision_b, pose_ab = (x.to(device, non_blocking=True) for x in batch)

            _, _, pose_pred = model(
                img_a, img_b,
                pose_vision_a=vision_a[..., :3],
                pose_vision_b=vision_b[..., :3],
            )

            errors, _, _ = geometric_translation_error(vision_a, vision_b, pose_ab, pose_pred)
            all_errors.extend(errors)

    print(f"Samples evaluated: {len(all_errors)}")
    print(f"Mean error:   {np.mean(all_errors):.2f}°")
    print(f"Median error: {np.median(all_errors):.2f}°")

    return all_errors


def test_pred_corr_gt_yaw(model, val_loader, device):
    model.eval()
    all_errors = []

    with torch.no_grad():
        for batch in val_loader:
            img_a, vision_a, img_b, vision_b, pose_ab = (x.to(device, non_blocking=True) for x in batch)

            cam_a, cam_b, patches_a, patches_b = model.backbone(img_a, img_b)
            corr_a = model.corr_head(patches_a)
            corr_b = model.corr_head(patches_b)

            B, P, D = corr_a.shape
            gh = model.pose_head.grid_h
            gw = model.pose_head.grid_w

            a = corr_a.reshape(B, gh, gw, D).mean(dim=1)
            b = corr_b.reshape(B, gh, gw, D).mean(dim=1)
            a = F.normalize(a, dim=-1)
            b = F.normalize(b, dim=-1)

            sim = torch.matmul(a, b.transpose(1, 2)) / model.pose_head.temperature
            prob_ab = F.softmax(sim, dim=-1)

            # GT depth
            depth_a, valid_a = model.pose_head._column_depth(vision_a[..., :3])
            depth_b, valid_b = model.pose_head._column_depth(vision_b[..., :3])
            depth_a = depth_a.to(prob_ab.dtype)
            depth_b = depth_b.to(prob_ab.dtype)
            valid_a = valid_a.to(prob_ab.dtype)
            valid_b = valid_b.to(prob_ab.dtype)

            theta = model.pose_head._column_angles(prob_ab.device, prob_ab.dtype)
            cos_theta = torch.cos(theta).unsqueeze(0)
            sin_theta = torch.sin(theta).unsqueeze(0)

            points_a = torch.stack([depth_a * cos_theta, depth_a * sin_theta], dim=-1)
            points_b = torch.stack([depth_b * cos_theta, depth_b * sin_theta], dim=-1)

            weights_ab = prob_ab * valid_b.unsqueeze(1)
            weights_ab = weights_ab / (weights_ab.sum(dim=-1, keepdim=True) + 1e-6)
            matched_points_b = torch.matmul(weights_ab, points_b)

            # GT yaw
            yaw = F.normalize(pose_ab[:, 2:], dim=-1)
            sin_yaw = yaw[:, 0]
            cos_yaw = yaw[:, 1]

            ax = points_a[..., 0]
            ay = points_a[..., 1]

            rotated_a = torch.stack([
                cos_yaw[:, None] * ax - sin_yaw[:, None] * ay,
                sin_yaw[:, None] * ax + cos_yaw[:, None] * ay,
            ], dim=-1)

            translation_per_col = matched_points_b - rotated_a

            match_mass = (prob_ab * valid_b.unsqueeze(1)).sum(dim=-1)
            confidence = prob_ab.max(dim=-1).values
            weights = valid_a * match_mass * confidence

            t_pred = (translation_per_col * weights.unsqueeze(-1)).sum(dim=1)
            t_pred = t_pred / (weights.sum(dim=1, keepdim=True) + 1e-6)
            t_pred = F.normalize(t_pred, dim=-1)

            t_gt = F.normalize(pose_ab[:, :2], dim=-1)
            cos_sim = torch.clamp((t_pred * t_gt).sum(dim=-1), -1.0, 1.0)
            errors = torch.rad2deg(torch.acos(cos_sim))

            valid = weights.sum(dim=1) > 1e-6
            all_errors.extend(errors[valid].cpu().tolist())

    print("TEST 2: GT depth + predicted correspondence + GT yaw")
    print(f"Samples evaluated: {len(all_errors)}")
    print(f"Mean error:   {np.mean(all_errors):.2f}°")
    print(f"Median error: {np.median(all_errors):.2f}°")

    return all_errors



def test_pred_depth_gt_corr_gt_yaw(model, val_loader, device):
    model.eval()
    all_errors = []

    with torch.no_grad():
        for batch in val_loader:
            img_a, vision_a, img_b, vision_b, pose_ab = (x.to(device, non_blocking=True) for x in batch)

            pred_vision_a, pred_vision_b, _ = model(
                img_a, img_b,
                pose_vision_a=vision_a[..., :3],
                pose_vision_b=vision_b[..., :3],
            )

            B, N, _ = vision_a.shape
            fov = torch.tensor(90.0 * torch.pi / 180.0, device=device, dtype=vision_a.dtype)
            theta = torch.linspace(-0.5 * fov, 0.5 * fov, N, device=device, dtype=vision_a.dtype)

            pred_pts_a = torch.stack([
                pred_vision_a[..., 2] * torch.cos(theta),
                pred_vision_a[..., 2] * torch.sin(theta),
            ], dim=-1)

            pred_pts_b = torch.stack([
                pred_vision_b[..., 2] * torch.cos(theta),
                pred_vision_b[..., 2] * torch.sin(theta),
            ], dim=-1)

            for b in range(B):
                occ_a = vision_a[b, :, 0] > 0.5
                occ_b = vision_b[b, :, 0] > 0.5
                ids_a = vision_a[b, :, 3].round().long()
                ids_b = vision_b[b, :, 3].round().long()

                yaw = F.normalize(pose_ab[b, 2:], dim=0)
                s, c = yaw[0], yaw[1]
                R = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])

                translations = []

                for cid in torch.unique(ids_a[occ_a]):
                    ia = torch.where(occ_a & (ids_a == cid))[0]
                    ib = torch.where(occ_b & (ids_b == cid))[0]

                    if ia.numel() != 1 or ib.numel() != 1:
                        continue

                    pa = pred_pts_a[b, ia[0]]
                    pb = pred_pts_b[b, ib[0]]
                    translations.append(pb - R @ pa)

                if not translations:
                    continue

                t_pred = F.normalize(torch.stack(translations).median(dim=0).values, dim=0)
                t_gt = F.normalize(pose_ab[b, :2], dim=0)

                cos_sim = torch.clamp(torch.dot(t_pred, t_gt), -1.0, 1.0)
                error = torch.rad2deg(torch.acos(cos_sim))
                all_errors.append(error.item())

    print("TEST 3: predicted depth + GT correspondence + GT yaw")
    print(f"Samples evaluated: {len(all_errors)}")
    print(f"Mean error:   {np.mean(all_errors):.2f}°")
    print(f"Median error: {np.median(all_errors):.2f}°")

    return all_errors

def _debug_rigid_transform_2d(A, B):
    ca = A.mean(dim=0)
    cb = B.mean(dim=0)
    Ac = A - ca
    Bc = B - cb
    dot = (Ac[:, 0] * Bc[:, 0] + Ac[:, 1] * Bc[:, 1]).sum()
    cross = (Ac[:, 0] * Bc[:, 1] - Ac[:, 1] * Bc[:, 0]).sum()
    norm = torch.sqrt(dot.square() + cross.square() + 1e-8)
    c = dot / norm
    s = cross / norm
    R = torch.stack([
        torch.stack([c, -s]),
        torch.stack([s, c]),
    ])
    t = cb - R @ ca
    return R, t, s, c


def _debug_ransac_rigid_2d(A, B, num_iters=200, inlier_threshold=0.4):
    n = A.shape[0]
    if n < 2:
        return None

    best_inliers = None
    best_count = 0
    best_error = float("inf")

    for _ in range(num_iters):
        idx = torch.randperm(n, device=A.device)[:2]
        R, t, _, _ = _debug_rigid_transform_2d(A[idx], B[idx])
        pred_B = A @ R.T + t
        residuals = torch.linalg.vector_norm(pred_B - B, dim=-1)
        inliers = residuals < inlier_threshold
        count = inliers.sum().item()
        if count < 2:
            continue
        mean_error = residuals[inliers].mean().item()
        if count > best_count or (count == best_count and mean_error < best_error):
            best_count = count
            best_error = mean_error
            best_inliers = inliers

    if best_inliers is None or best_inliers.sum() < 2:
        return None

    R, t, s, c = _debug_rigid_transform_2d(A[best_inliers], B[best_inliers])
    return R, t, s, c, best_inliers


def test_ransac_geometry_ablation_gt_coarse(
    model,
    loader,
    device,
    search_radius=16,
    num_iters=200,
    inlier_threshold=0.4,
    gt_occ_thresh=0.5,
    pred_occ_thresh=0.05,
    fov_degrees=90.0,
):
    """
    Isolerar den predikterade cylindergeometrin med perfekt coarse correspondence.

    GT cylinder-ID används ENDAST för att ange korrekt coarse center i B.
    Fine matching använder radius från vald konfiguration och RANSAC använder
    punktgeometri från vald depth-konfiguration.

    Konfigurationer:
      - All GT
      - Pred depth only
      - Pred radius only
      - Pred occupancy only
      - All predicted
    """
    model.eval()

    configs = {
        "All GT": (True, True, True),
        "Pred depth only": (True, True, False),
        "Pred radius only": (True, False, True),
        "Pred occupancy only": (False, True, True),
        "All predicted": (False, False, False),
    }

    results = {
        name: {"t": [], "yaw": [], "matches": [], "inliers": [], "zeros": 0}
        for name in configs
    }

    with torch.no_grad():
        for batch in loader:
            img_a, gt_a, img_b, gt_b, pose_gt = (
                x.to(device, non_blocking=True) for x in batch
            )

            _, _, patches_a, patches_b = model.backbone(img_a, img_b)
            pred_a = model.vision_head(patches_a)
            pred_b = model.vision_head(patches_b)

            B, N, _ = gt_a.shape
            dtype = pred_a.dtype
            dev = pred_a.device

            fov = torch.tensor(
                float(fov_degrees) * torch.pi / 180.0,
                device=dev,
                dtype=dtype,
            )
            theta = torch.linspace(
                -0.5 * fov,
                0.5 * fov,
                N,
                device=dev,
                dtype=dtype,
            )
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)

            for name, (use_gt_occ, use_gt_radius, use_gt_depth) in configs.items():
                occ_a_values = gt_a[..., 0] if use_gt_occ else pred_a[..., 0]
                occ_b_values = gt_b[..., 0] if use_gt_occ else pred_b[..., 0]
                occ_threshold = gt_occ_thresh if use_gt_occ else pred_occ_thresh

                radius_a_values = gt_a[..., 1] if use_gt_radius else pred_a[..., 1]
                radius_b_values = gt_b[..., 1] if use_gt_radius else pred_b[..., 1]
                depth_a_values = gt_a[..., 2] if use_gt_depth else pred_a[..., 2]
                depth_b_values = gt_b[..., 2] if use_gt_depth else pred_b[..., 2]

                pts_a = torch.stack([
                    depth_a_values * cos_theta,
                    depth_a_values * sin_theta,
                ], dim=-1)
                pts_b = torch.stack([
                    depth_b_values * cos_theta,
                    depth_b_values * sin_theta,
                ], dim=-1)

                for b in range(B):
                    gt_occ_a = gt_a[b, :, 0] > gt_occ_thresh
                    gt_occ_b = gt_b[b, :, 0] > gt_occ_thresh
                    active_a = occ_a_values[b] > occ_threshold
                    active_b = occ_b_values[b] > occ_threshold
                    ids_a = gt_a[b, :, 3].round().long()
                    ids_b = gt_b[b, :, 3].round().long()

                    matched_a = []
                    matched_b = []

                    for ia_gt in torch.where(gt_occ_a)[0]:
                        same_id_b = torch.where(gt_occ_b & (ids_b == ids_a[ia_gt]))[0]
                        if same_id_b.numel() != 1:
                            continue

                        # Perfect coarse correspondence: true B bin is only the search center.
                        ib_gt = same_id_b[0]

                        # If occupancy is predicted, it decides whether the A cylinder is active.
                        if not active_a[ia_gt]:
                            continue

                        center = ib_gt.item()
                        lo = max(0, center - search_radius)
                        hi = min(N, center + search_radius + 1)
                        candidates = torch.arange(lo, hi, device=dev)
                        candidates = candidates[active_b[candidates]]
                        if candidates.numel() == 0:
                            continue

                        # Fine matching uses GT or predicted radius according to config.
                        radius_a = radius_a_values[b, ia_gt]
                        radius_diff = torch.abs(radius_b_values[b, candidates] - radius_a)
                        ib = candidates[radius_diff.argmin()]

                        matched_a.append(pts_a[b, ia_gt])
                        matched_b.append(pts_b[b, ib])

                    if len(matched_a) < 2:
                        results[name]["zeros"] += 1
                        continue

                    A = torch.stack(matched_a)
                    Bpts = torch.stack(matched_b)
                    ransac = _debug_ransac_rigid_2d(
                        A,
                        Bpts,
                        num_iters=num_iters,
                        inlier_threshold=inlier_threshold,
                    )
                    if ransac is None:
                        results[name]["zeros"] += 1
                        continue

                    R, t, s, c, inliers = ransac
                    t_pred = F.normalize(t, dim=0)
                    t_gt = F.normalize(pose_gt[b, :2], dim=0)
                    t_cos = torch.clamp(torch.dot(t_pred, t_gt), -1.0, 1.0)
                    t_error = torch.rad2deg(torch.acos(t_cos)).item()

                    yaw_pred = torch.stack([s, c])
                    yaw_gt = F.normalize(pose_gt[b, 2:], dim=0)
                    yaw_cos = torch.clamp(torch.dot(yaw_pred, yaw_gt), -1.0, 1.0)
                    yaw_error = torch.rad2deg(torch.acos(yaw_cos)).item()

                    results[name]["t"].append(t_error)
                    results[name]["yaw"].append(yaw_error)
                    results[name]["matches"].append(A.shape[0])
                    results[name]["inliers"].append(inliers.sum().item())

    print("RANSAC GEOMETRY ABLATION - GT COARSE")
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

    for name in configs:
        r = results[name]
        if len(r["t"]) == 0:
            print(f"{name:<24}{'NO VALID SAMPLES':>40}{r['zeros']:>8}")
            continue
        print(
            f"{name:<24}"
            f"{np.mean(r['t']):>9.2f}"
            f"{np.median(r['t']):>9.2f}"
            f"{np.percentile(r['t'], 90):>9.2f}"
            f"{np.mean(r['yaw']):>10.2f}"
            f"{np.median(r['yaw']):>9.2f}"
            f"{np.mean(r['matches']):>10.1f}"
            f"{np.mean(r['inliers']):>10.1f}"
            f"{r['zeros']:>8}"
        )

    return results
