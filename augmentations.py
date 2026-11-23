"""Image augmentations for pose_from_cylinders.

Photometric augmentations only: they change image appearance without changing
camera geometry, so the existing pose/vision labels remain valid.
"""

from torchvision import transforms


def build_train_transform(image_size: int = 128) -> transforms.Compose:
    """Build the augmentation pipeline used for training images."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            # Simulate more noticeable defocus / motion softness.
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=7, sigma=(0.2, 2.0))],
                p=0.35,
            ),
            # Stronger random changes to brightness, contrast, saturation and hue.
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.40,
                        contrast=0.40,
                        saturation=0.40,
                        hue=0.08,
                    )
                ],
                p=0.80,
            ),
            # More often remove color information entirely.
            transforms.RandomGrayscale(p=0.25),
            transforms.ToTensor(),
        ]
    )


def build_eval_transform(image_size: int = 128) -> transforms.Compose:
    """Build the deterministic preprocessing pipeline used for validation/test."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
