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
            # Mild blur: occasionally simulate defocus / motion softness.
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))],
                p=0.20,
            ),
            # Randomly change brightness, contrast and saturation.
            # Hue is kept off initially because it is often more aggressive.
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.20,
                        contrast=0.20,
                        saturation=0.20,
                        hue=0.0,
                    )
                ],
                p=0.50,
            ),
            # Occasionally remove most/all color information.
            transforms.RandomGrayscale(p=0.15),
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
