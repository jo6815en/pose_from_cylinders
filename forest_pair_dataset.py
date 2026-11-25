import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class ForestPairDataset(Dataset):
    """Image-pair dataset for forest sequences and their relative camera poses.

    Expected layout::

        dataset_dir/
            images/
            relative_poses.json  # or another file selected by poses_file

    Each pose JSON is a list of records with ``from_image``, ``to_image``, and
    either ``T_a_to_b`` or ``T_ab``. The returned transform maps points from
    camera A coordinates to camera B coordinates.
    """

    def __init__(
        self,
        dataset_dir,
        transform=None,
        return_names=False,
        poses_file="relative_poses.json",
    ):
        self.dataset_dir = Path(dataset_dir)
        self.image_dir = self.dataset_dir / "images"
        self.return_names = return_names
        self.poses_file = poses_file

        poses_path = self.dataset_dir / poses_file
        with poses_path.open("r", encoding="utf-8") as file:
            self.pairs = json.load(file)

        if not isinstance(self.pairs, list):
            raise ValueError(f"Expected a list of image pairs in {poses_path}")

        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        image_a_name = item.get("from_image", item.get("image_a"))
        image_b_name = item.get("to_image", item.get("image_b"))
        if image_a_name is None or image_b_name is None:
            raise KeyError(
                "Pose item must contain 'from_image'/'to_image' or 'image_a'/'image_b'"
            )

        image_a_path = self.image_dir / image_a_name
        image_b_path = self.image_dir / image_b_name
        with Image.open(image_a_path) as image:
            image_a = self.transform(image.convert("RGB"))
        with Image.open(image_b_path) as image:
            image_b = self.transform(image.convert("RGB"))

        if "T_a_to_b" in item:
            pose = item["T_a_to_b"]
        elif "T_ab" in item:
            pose = item["T_ab"]
        else:
            raise KeyError("Pose item must contain 'T_a_to_b' or 'T_ab'")

        T_ab = torch.tensor(pose, dtype=torch.float32)
        if T_ab.shape != (4, 4):
            raise ValueError(
                f"Relative pose for pair {idx} must have shape [4, 4], "
                f"got {tuple(T_ab.shape)}"
            )

        if self.return_names:
            return {
                "image_a": image_a,
                "image_b": image_b,
                "T_ab": T_ab,
                "image_a_name": image_a_name,
                "image_b_name": image_b_name,
            }
        return image_a, image_b, T_ab
