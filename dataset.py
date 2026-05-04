from pathlib import Path
import random
import json
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SceneTwoPairsDataset(Dataset):
    def __init__(
        self,
        root_dir="dataset",
        image_size=128,
        scale_xy=2.8, 
        return_paths=False,
    ):
        self.root_dir = Path(root_dir)
        self.return_paths = return_paths
        self.scale_xy = scale_xy

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        self.scenes = []

        for scene_dir in sorted(self.root_dir.glob("scene_*")):
            label_file = scene_dir / "labels.json"
            if not label_file.exists():
                continue

            with open(label_file, "r") as f:
                data = json.load(f)

            pairs = data.get("pairs", [])
            if len(pairs) >= 1:
                self.scenes.append(data)

        if not self.scenes:
            raise ValueError(f"Inga scener hittades i {root_dir}")

        self.length = len(self.scenes)

    def __len__(self):
        return self.length

    def _load_image(self, path):
        return self.transform(Image.open(path).convert("RGB"))

    def _load_vision(self, vision):
        return torch.tensor(vision, dtype=torch.float32)

    def _camera_to_matrix(self, camera):
        """
        Bygger world->camera extrinsic-matris från R och t.
        Förutsätter att camera["R"] och camera["t"] finns.
        """
        R = torch.tensor(camera["R"], dtype=torch.float32)
        t = torch.tensor(camera["t"], dtype=torch.float32).view(3)

        T = torch.eye(4, dtype=torch.float32)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def _rotation_matrix_to_quaternion(self, R):
        """
        Returnerar quaternion som [qw, qx, qy, qz].
        """
        trace = R[0, 0] + R[1, 1] + R[2, 2]

        if trace > 0.0:
            s = torch.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s

        q = torch.stack([qw, qx, qy, qz])
        q = q / (torch.linalg.norm(q) + 1e-8)

        # Gör representationen unik-ish
        if q[0] < 0:
            q = -q

        return q
    
    def _yaw_from_rotation_matrix(self, R):
        """
        Antag z-up: yaw runt z-axeln.
        Returnerar vinkel i radianer.
        """
        return torch.atan2(R[1, 0], R[0, 0])

    def _relative_pose(self, camera1, camera2):
        """
        Returnerar relativ pose som:
        [tx, ty, tz, qw, qx, qy, qz]
        för transformen från camera1 till camera2.

        Förutsätter world->camera extrinsics.
        """
        T1 = self._camera_to_matrix(camera1)
        T2 = self._camera_to_matrix(camera2)

        # relativ transform från 1 till 2
        T_rel = T2 @ torch.linalg.inv(T1)

        t = T_rel[:3, 3]
        R = T_rel[:3, :3]
        q = self._rotation_matrix_to_quaternion(R)

        return torch.cat([t, q], dim=0)
    
    def _relative_pose_2d(self, camera1, camera2):
        """
        Returnerar:
        [tx, ty, sin(yaw), cos(yaw)]
        """

        T1 = self._camera_to_matrix(camera1)
        T2 = self._camera_to_matrix(camera2)

        T_rel = T2 @ torch.linalg.inv(T1)

        # XY translation + normalisering
        t = T_rel[:2, 3] / self.scale_xy

        # yaw
        R = T_rel[:3, :3]
        yaw = torch.atan2(R[1, 0], R[0, 0])

        pose = torch.tensor(
            [t[0], t[1], torch.sin(yaw), torch.cos(yaw)],
            dtype=torch.float32,
        )
        return pose

    def __getitem__(self, idx):
        scene = self.scenes[idx % len(self.scenes)]
        pairs = scene["pairs"]

        # välj ett par från scenen
        p = random.choice(pairs)

        path_a = p["image1"]
        path_b = p["image2"]

        img_a = self._load_image(path_a)
        img_b = self._load_image(path_b)

        # vision target för första bilden
        vision_a = self._load_vision(p["vision1"])
        vision_b = self._load_vision(p["vision2"])

        # cameras kan ligga i pair eller på scene-nivå
        camera1 = p.get("camera1", scene.get("camera1"))
        camera2 = p.get("camera2", scene.get("camera2"))

        if camera1 is None or camera2 is None:
            raise KeyError(
                "Saknar camera1/camera2 i pair eller scene. "
                "Lägg in dem i labels.json."
            )

        # pose_ab = self._relative_pose(camera1, camera2)
        pose_ab = self._relative_pose_2d(camera1, camera2)

        if self.return_paths:
            return (
                img_a, vision_a,
                img_b, vision_b,
                pose_ab,
                path_a, path_b
            )

        return (
            img_a, vision_a,
            img_b, vision_b,
            pose_ab
        )