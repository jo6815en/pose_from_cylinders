from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass(frozen=True)
class _Sample:
    scene_dir: Path
    pair: Dict[str, Any]
    scene_camera1: Optional[Dict[str, Any]]
    scene_camera2: Optional[Dict[str, Any]]


class SceneTwoPairsDataset(Dataset):
    """Dataset for paired cylinder scenes.

    Each dataset item corresponds to one pair inside one scene.
    This avoids the old behavior where __getitem__ randomly sampled a pair
    from a scene, which made epoch length and sampling non-deterministic.
    """

    def __init__(
        self,
        root_dir: str | Path = "dataset",
        image_size: int = 128,
        return_paths: bool = False,
        shuffle_pairs: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.return_paths = return_paths
        self.shuffle_pairs = shuffle_pairs

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

        self.samples: List[_Sample] = []
        for scene_dir in sorted(self.root_dir.glob("scene_*")):
            label_file = scene_dir / "labels.json"
            if not label_file.exists():
                continue

            with open(label_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            pairs = data.get("pairs", [])
            if not isinstance(pairs, list) or not pairs:
                continue

            scene_camera1 = data.get("camera1")
            scene_camera2 = data.get("camera2")

            for pair in pairs:
                if not isinstance(pair, dict):
                    continue
                if "image1" not in pair or "image2" not in pair:
                    continue
                if "vision1" not in pair or "vision2" not in pair:
                    continue
                self.samples.append(
                    _Sample(
                        scene_dir=scene_dir,
                        pair=pair,
                        scene_camera1=scene_camera1,
                        scene_camera2=scene_camera2,
                    )
                )

        if not self.samples:
            raise ValueError(f"Inga giltiga par hittades i {self.root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_path(self, scene_dir: Path, maybe_path: Any) -> Path:
        path = Path(maybe_path)
        if path.is_absolute():
            return path

        candidate = scene_dir / path
        if candidate.exists():
            return candidate

        # Fall back to the raw relative path so callers get a useful error later.
        return path

    def _load_image(self, path: Path) -> torch.Tensor:
        with Image.open(path) as img:
            img = img.convert("RGB")
            return self.transform(img)

    def _load_vision(self, vision: Any) -> torch.Tensor:
        return torch.as_tensor(vision, dtype=torch.float32)

    def _camera_to_matrix(self, camera: Dict[str, Any]) -> torch.Tensor:
        if "R" not in camera or "t" not in camera:
            raise KeyError('camera måste innehålla nycklarna "R" och "t"')

        R = torch.as_tensor(camera["R"], dtype=torch.float32)
        t = torch.as_tensor(camera["t"], dtype=torch.float32).reshape(3)

        if R.shape != (3, 3):
            raise ValueError(f"Förväntade R med shape (3, 3), fick {tuple(R.shape)}")

        T = torch.eye(4, dtype=torch.float32)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def _rotation_matrix_to_quaternion(self, R: torch.Tensor) -> torch.Tensor:
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

        # Make the representation deterministic up to sign.
        if q[0] < 0:
            q = -q
        return q

    def _relative_pose(self, camera1: Dict[str, Any], camera2: Dict[str, Any]) -> torch.Tensor:
        T1 = self._camera_to_matrix(camera1)
        T2 = self._camera_to_matrix(camera2)
        T_rel = T2 @ torch.linalg.inv(T1)
        t = T_rel[:3, 3]
        q = self._rotation_matrix_to_quaternion(T_rel[:3, :3])
        return torch.cat([t, q], dim=0)

    def _relative_pose_2d(self, camera1: Dict[str, Any], camera2: Dict[str, Any]) -> torch.Tensor:
        T1 = self._camera_to_matrix(camera1)
        T2 = self._camera_to_matrix(camera2)
        T_rel = T2 @ torch.linalg.inv(T1)

        t_xy = T_rel[:2, 3]
        R = T_rel[:3, :3]
        yaw = torch.atan2(R[1, 0], R[0, 0])

        return torch.tensor(
            [t_xy[0], t_xy[1], torch.sin(yaw), torch.cos(yaw)],
            dtype=torch.float32,
        )

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        pair = sample.pair
        scene_dir = sample.scene_dir

        path_a = self._resolve_path(scene_dir, pair["image1"])
        path_b = self._resolve_path(scene_dir, pair["image2"])

        img_a = self._load_image(path_a)
        img_b = self._load_image(path_b)

        vision_a = self._load_vision(pair["vision1"])
        vision_b = self._load_vision(pair["vision2"])

        camera1 = pair.get("camera1", sample.scene_camera1)
        camera2 = pair.get("camera2", sample.scene_camera2)

        if camera1 is None or camera2 is None:
            raise KeyError(
                "Saknar camera1/camera2 i pair eller scene. Lägg in dem i labels.json."
            )

        pose_ab = self._relative_pose_2d(camera1, camera2)

        if self.return_paths:
            return img_a, vision_a, img_b, vision_b, pose_ab, str(path_a), str(path_b), camera1, camera2

        return img_a, vision_a, img_b, vision_b, pose_ab
