from torch.utils.data import DataLoader
from torchvision import transforms

from colmap_pair_dataset import ColmapPairDataset

dataset = ColmapPairDataset(
    dataset_dir="forestdataset",
     image_size=128
     )

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
)

for batch in loader:
    image_a, image_b, T_ab = batch 

    print(image_a.shape, image_b.shape, T_ab.shape)
    break
