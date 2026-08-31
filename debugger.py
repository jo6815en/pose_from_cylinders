from pathlib import Path
import torch
import math
from dataset import SceneTwoPairsDataset

def angles_from_R(R):
    return {
        "z_yaw": torch.atan2(R[1, 0], R[0, 0]),
        "y_yaw": torch.atan2(R[0, 2], R[0, 0]),
        "x_roll": torch.atan2(R[2, 1], R[1, 1]),
    }


def yaw_from_R(R):
    return torch.atan2(R[1, 0], R[0, 0])


def yaw_from_Trel(T_rel):
    R = T_rel[:3, :3]
    return torch.atan2(R[1, 0], R[0, 0])


def camera_to_T(cam):
    R = torch.tensor(cam["R"], dtype=torch.float32)
    t = torch.tensor(cam["t"], dtype=torch.float32)
    T = torch.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def check_yaw(cam1, cam2):
    R1 = torch.tensor(cam1["R"], dtype=torch.float32)
    R2 = torch.tensor(cam2["R"], dtype=torch.float32)

    yaw1 = yaw_from_R(R1)
    yaw2 = yaw_from_R(R2)

    T1 = camera_to_T(cam1)
    T2 = camera_to_T(cam2)
    T_rel = T2 @ torch.linalg.inv(T1)

    yaw_rel = yaw_from_Trel(T_rel)

    print("yaw1 rad:", yaw1.item())
    print("yaw1 deg:", yaw1.item() * 180 / math.pi)

    print("yaw2 rad:", yaw2.item())
    print("yaw2 deg:", yaw2.item() * 180 / math.pi)

    print("relative yaw rad:", yaw_rel.item())
    print("relative yaw deg:", yaw_rel.item() * 180 / math.pi)

    print("manual yaw2 - yaw1 deg:", (yaw2 - yaw1).item() * 180 / math.pi)

    x_axis = T_rel[:2, 0]
    print("projected x-axis:", x_axis)

    return yaw1, yaw2, yaw_rel, T_rel


def bug_check(idx=0):
    dataset = SceneTwoPairsDataset("dataset", debugg=True)
    sample = dataset[idx]

    img_a, vision_a, img_b, pose, path_a, path_b, camera1, camera2 = sample
    raw_pair = dataset.samples[idx].pair
    scene_dir = dataset.samples[idx].scene_dir

    print("=== SAMPLE IDENTITY ===")
    print("scene_dir:", scene_dir)
    print("pair_id:", raw_pair.get("pair_id"))
    print("image1 path:", raw_pair.get("image1"))
    print("image2 path:", raw_pair.get("image2"))

    print("\n=== RAW PAIR FROM LOADED JSON ===")
    print(raw_pair["camera2"])

    print("\n=== RETURNED camera2 ===")
    print(camera2)

    print("\n=== PATHS ===")
    print(path_a)
    print(path_b)

    print("\n=== YAW ===")
    check_yaw(camera1, camera2)

    R1 = torch.tensor(camera1["R"], dtype=torch.float32)
    R2 = torch.tensor(camera2["R"], dtype=torch.float32)

    print("camera1 angles:", {k: v.item() for k, v in angles_from_R(R1).items()})
    print("camera2 angles:", {k: v.item() for k, v in angles_from_R(R2).items()})

    T1 = camera_to_T(camera1)
    T2 = camera_to_T(camera2)
    R_rel = (T2 @ torch.linalg.inv(T1))[:3, :3]

    print("relative angles:", {k: v.item() for k, v in angles_from_R(R_rel).items()})



from losses import (
    match_sinkhorn_between_views,
    matched_radius_consistency_loss,
    matched_reprojection_loss_2d,
)
from torch.utils.data import DataLoader
from dataset import SceneTwoPairsDataset


def _isin_1d(values, candidates):
    if values.numel() == 0 or candidates.numel() == 0:
        return torch.zeros_like(values, dtype=torch.bool)
    return (values[:, None] == candidates[None, :]).any(dim=1)

def debug_sinkhorn_matching(device="cuda"):
    SINKHORN_DEBUG_BATCH_SIZE = 4
    SINKHORN_OCC_THRESH = 0.5
    SINKHORN_TEMPERATURE = 0.05
    SINKHORN_ITERS = 100
    SINKHORN_DUSTBIN_COST = 1.0
    SINKHORN_MIN_MATCH_PROB = 0.0

    debug_device = torch.device(device if "device" in globals() else "cuda" if torch.cuda.is_available() else "cpu")

    debug_dataset = SceneTwoPairsDataset(
        root_dir="dataset",
        return_two_pairs=False,
    )

    debug_loader = DataLoader(
        debug_dataset,
        batch_size=SINKHORN_DEBUG_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    img_a, vision_a, img_b, vision_b, pose_ab = next(iter(debug_loader))
    vision_a = vision_a.to(debug_device)
    vision_b = vision_b.to(debug_device)
    pose_ab = pose_ab.to(debug_device)

    with torch.no_grad():
        matches = match_sinkhorn_between_views(
            vision_a,
            vision_b,
            pose_ab,
            occ_thresh=SINKHORN_OCC_THRESH,
            temperature=SINKHORN_TEMPERATURE,
            sinkhorn_iters=SINKHORN_ITERS,
            dustbin_cost=SINKHORN_DUSTBIN_COST,
            min_match_prob=SINKHORN_MIN_MATCH_PROB,
        )

        radius_sink = matched_radius_consistency_loss(
            vision_a,
            pred_vision_b=vision_b,
            relative_pose_pred=pose_ab,
            matching_mode="sinkhorn",
            occ_thresh=SINKHORN_OCC_THRESH,
        )
        reproj_sink = matched_reprojection_loss_2d(
            vision_a,
            pred_vision_b=vision_b,
            relative_pose_pred=pose_ab,
            matching_mode="sinkhorn",
            occ_thresh=SINKHORN_OCC_THRESH,
            fov_degrees=90.0,
        )
        radius_gt = matched_radius_consistency_loss(
            vision_a,
            vision_a,
            vision_b,
            vision_b,
            relative_pose_pred=pose_ab,
            matching_mode="gt",
            occ_thresh=SINKHORN_OCC_THRESH,
        )
        reproj_gt = matched_reprojection_loss_2d(
            vision_a,
            vision_a,
            vision_b,
            vision_b,
            relative_pose_pred=pose_ab,
            matching_mode="gt",
            occ_thresh=SINKHORN_OCC_THRESH,
            fov_degrees=90.0,
        )

    total_matches = 0
    total_correct = 0
    total_matchable_a = 0
    bad_examples = []

    print("=== Sinkhorn-check med ground truth vision + ground truth pose ===")
    print(f"batch={vision_a.shape[0]}, bins={vision_a.shape[1]}, device={debug_device}")
    print(f"radius loss: sinkhorn={radius_sink.item():.6f} | gt-id={radius_gt.item():.6f}")
    print(f"reproj loss: sinkhorn={reproj_sink.item():.6f} | gt-id={reproj_gt.item():.6f}")

    for batch_i, match in enumerate(matches):
        occ_a = vision_a[batch_i, :, 0] > SINKHORN_OCC_THRESH
        occ_b = vision_b[batch_i, :, 0] > SINKHORN_OCC_THRESH

        ids_occ_a = vision_a[batch_i, occ_a, 3].long()
        ids_occ_b = vision_b[batch_i, occ_b, 3].long()
        ids_a_unique = ids_occ_a.unique()
        ids_b_unique = ids_occ_b.unique()
        common_ids = ids_a_unique[_isin_1d(ids_a_unique, ids_b_unique)]

        idx_a = match["indices_a"]
        idx_b = match["matched_indices_b"]
        ids_a = vision_a[batch_i, idx_a, 3].long()
        ids_b = vision_b[batch_i, idx_b, 3].long()

        same_id = ids_a == ids_b
        a_id_is_visible_in_b = _isin_1d(ids_a, common_ids)
        correct = same_id & a_id_is_visible_in_b

        n_matches = idx_a.numel()
        n_correct = correct.sum().item()
        n_matchable_a = _isin_1d(ids_occ_a, common_ids).sum().item()

        total_matches += n_matches
        total_correct += n_correct
        total_matchable_a += n_matchable_a

        precision = n_correct / max(n_matches, 1)
        recall = n_correct / max(n_matchable_a, 1)
        print(
            f"sample {batch_i}: "
            f"occ_a={occ_a.sum().item()} occ_b={occ_b.sum().item()} "
            f"common_ids={common_ids.detach().cpu().tolist()} "
            f"matches={n_matches} correct={n_correct} "
            f"precision={precision:.3f} recall={recall:.3f}"
        )

        wrong_pos = torch.where(~correct)[0][:5]
        for pos in wrong_pos:
            bad_examples.append({
                "sample": batch_i,
                "a_bin": int(idx_a[pos].item()),
                "b_bin": int(idx_b[pos].item()),
                "id_a": int(ids_a[pos].item()),
                "id_b": int(ids_b[pos].item()),
                "cost": float(match["matched_cost"][pos].item()),
                "prob": float(match["match_prob"][pos].item()),
            })

    precision = total_correct / max(total_matches, 1)
    recall = total_correct / max(total_matchable_a, 1)
    print(f"TOTAL: correct={total_correct}/{total_matches} precision={precision:.3f}")
    print(f"TOTAL: recall mot matchbara A-bins={total_correct}/{total_matchable_a} recall={recall:.3f}")

    if bad_examples:
        print("\nFörsta felmatchningar:")
        for ex in bad_examples[:10]:
            print(ex)
    else:
        print("\nInga felmatchningar i den här batchen.")

