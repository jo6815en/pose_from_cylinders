import torch
import torch.nn.functional as F
import numpy as np

def evaluate_128_correspondence_tolerance(model, loader, device, occ_thresh=0.5, temperature=0.1):
    model.eval()
    total = 0
    exact = 0
    within_1 = 0
    within_2 = 0
    within_4 = 0
    errors = []

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

                B = img_a.shape[0]

                for b in range(B):
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

                        ia = match_a[0]
                        ib = match_b[0]

                        error = abs(pred_b[ia].item() - ib.item())

                        total += 1
                        errors.append(error)

                        exact += error == 0
                        within_1 += error <= 1
                        within_2 += error <= 2
                        within_4 += error <= 4

    print("128-BIN CORRESPONDENCE TOLERANCE")
    print("----------------------------------------")
    print(f"Matches:       {total}")
    print(f"Exact:         {100 * exact / total:.1f}%")
    print(f"Within ±1:     {100 * within_1 / total:.1f}%")
    print(f"Within ±2:     {100 * within_2 / total:.1f}%")
    print(f"Within ±4:     {100 * within_4 / total:.1f}%")
    print(f"Mean error:    {np.mean(errors):.2f} bins")
    print(f"Median error:  {np.median(errors):.2f} bins")


def evaluate_128_correspondences(model, loader, device, occ_thresh=0.5, temperature=0.1):
    model.eval()

    total_matches = 0
    top1_correct = 0
    total_bin_error = 0.0
    total_correct_prob = 0.0
    total_loss = 0.0
    loss_count = 0

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
                    prob = F.softmax(sim, dim=-1)

                    for cid in ids_a.unique():
                        match_a = idx_a[ids_a == cid]
                        match_b = idx_b[ids_b == cid]

                        if match_a.numel() != 1 or match_b.numel() != 1:
                            continue

                        ia = match_a[0]
                        ib = match_b[0]

                        pred_ib = prob[ia].argmax()

                        total_matches += 1
                        top1_correct += int(pred_ib == ib)
                        total_bin_error += abs(pred_ib.item() - ib.item())
                        total_correct_prob += prob[ia, ib].item()

                        total_loss += F.cross_entropy(
                            sim[ia].unsqueeze(0),
                            ib.view(1),
                        ).item()
                        loss_count += 1

        top1 = top1_correct / max(total_matches, 1)
        mean_bin_error = total_bin_error / max(total_matches, 1)
        correct_prob = total_correct_prob / max(total_matches, 1)
        mean_loss = total_loss / max(loss_count, 1)
        uniform = 1.0 / 128.0
        lift = correct_prob / uniform

        print("----------------------------------------")
        print("128-BIN CORRESPONDENCE")
        print("----------------------------------------")
        print(f"Correspondence loss: {mean_loss:.4f}")
        print(f"Matches:             {total_matches}")
        print(f"Top-1 hit rate:      {100 * top1:.1f}%")
        print(f"Mean bin error:      {mean_bin_error:.2f}")
        print(f"Correct probability: {correct_prob:.3f}")
        print(f"Uniform baseline:    {uniform:.3f}")
        print(f"Lift:                {lift:.2f}x")

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