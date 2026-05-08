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
