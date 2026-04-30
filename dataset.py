from pathlib import Path
import random
import json
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


class SceneTwoPairsDataset(Dataset):
    def __init__(
        self,
        root_dir="dataset",
        image_size=128,
        pairs_per_scene=10,
        return_paths=False,
    ):
        self.root_dir = Path(root_dir)
        self.return_paths = return_paths
        self.pairs_per_scene = pairs_per_scene

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

            pairs = data["pairs"]

            if len(pairs) >= 2:
                self.scenes.append(pairs)

        if not self.scenes:
            raise ValueError(f"Inga scener hittades i {root_dir}")

        self.length = len(self.scenes) * self.pairs_per_scene

    def __len__(self):
        return self.length

    def _load_image(self, path):
        return self.transform(Image.open(path).convert("RGB"))

    def __getitem__(self, idx):
        pairs = self.scenes[idx % len(self.scenes)]

        # välj två olika par
        p1, p2 = random.sample(pairs, 2)

        path_a1 = p1["image1"]
        path_b1 = p1["image2"]

        path_a2 = p2["image1"]
        path_b2 = p2["image2"]

        img_a1 = self._load_image(path_a1)
        img_b1 = self._load_image(path_b1)
        img_a2 = self._load_image(path_a2)
        img_b2 = self._load_image(path_b2)

        if self.return_paths:
            return img_a1, img_b1, img_a2, img_b2, path_a1, path_b1, path_a2, path_b2

        return img_a1, img_b1, img_a2, img_b2