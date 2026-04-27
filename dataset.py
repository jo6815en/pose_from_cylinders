from pathlib import Path
import random
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


class SceneTwoPairsDataset(Dataset):
    def __init__(
        self,
        root_dir="data",
        image_size=128,
        pairs_per_scene=20,
        return_paths=False,
    ):
        self.root_dir = Path(root_dir)
        self.return_paths = return_paths
        self.pairs_per_scene = pairs_per_scene

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        self.scenes = []  # list of [img_paths]
        for scene_dir in sorted(self.root_dir.glob("scene_*")):
            imgs = sorted((scene_dir / "images").glob("*.png"))
            if len(imgs) >= 3:  # behövs minst 3 bilder för två olika par
                self.scenes.append(imgs)

        if not self.scenes:
            raise ValueError(f"Inga scener med minst 3 bilder hittades i {root_dir}")

        self.length = len(self.scenes) * self.pairs_per_scene

    def __len__(self):
        return self.length

    def _load_image(self, path):
        return self.transform(Image.open(path).convert("RGB"))

    @staticmethod
    def _pair_from_index(n, k):
        i = 0
        while k >= (n - i - 1):
            k -= (n - i - 1)
            i += 1
        j = i + 1 + k
        return i, j

    def __getitem__(self, idx):
        imgs = self.scenes[idx % len(self.scenes)]
        n = len(imgs)
        num_pairs = n * (n - 1) // 2

        # Två olika par från samma scen
        k1, k2 = random.sample(range(num_pairs), 2)

        i1, j1 = self._pair_from_index(n, k1)
        i2, j2 = self._pair_from_index(n, k2)

        path_a1, path_b1 = imgs[i1], imgs[j1]
        path_a2, path_b2 = imgs[i2], imgs[j2]

        img_a1 = self._load_image(path_a1)
        img_b1 = self._load_image(path_b1)
        img_a2 = self._load_image(path_a2)
        img_b2 = self._load_image(path_b2)

        if self.return_paths:
            return img_a1, img_b1, img_a2, img_b2, path_a1, path_b1, path_a2, path_b2

        return img_a1, img_b1, img_a2, img_b2

        