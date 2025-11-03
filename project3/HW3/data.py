import math
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def _mnist_transform(resize_to_32: bool = True, normalize: bool = True):
    t = []
    t.append(transforms.ToTensor())
    if resize_to_32:
        # MNIST is 28x28; make it 32x32 to match the model requirement
        t.append(transforms.Resize((32, 32)))
    if normalize:
        # Normalize to [0,1] is already achieved by ToTensor; additional normalize is optional
        # Keep [0,1] for BCE; you can switch to mean/std if using MSE
        pass
    return transforms.Compose(t)


def get_mnist_dataloaders(
    data_dir: str,
    batch_size: int = 128,
    num_workers: int = 4,
    val_split: float = 0.0,
    download: bool = True,
    resize_to_32: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """
    Returns train, val (optional), and test dataloaders for MNIST.

    - Images are converted to 32x32 if resize_to_32 is True.
    - Pixel range is [0, 1], suitable for BCE loss.
    """

    transform = _mnist_transform(resize_to_32=resize_to_32)

    train_full = datasets.MNIST(root=data_dir, train=True, transform=transform, download=download)
    test = datasets.MNIST(root=data_dir, train=False, transform=transform, download=download)

    val_loader = None
    if val_split and 0.0 < val_split < 1.0:
        val_size = int(len(train_full) * val_split)
        train_size = len(train_full) - val_size
        train, val = random_split(train_full, [train_size, val_size])
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_full, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
