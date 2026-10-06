import os
import glob
from typing import Dict, Tuple, Optional
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets

# Standard ImageNet normalization parameters used by pretrained CNNs (MobileNetV2 / EfficientNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = (224, 224)
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

def get_train_transforms() -> transforms.Compose:
    """Data augmentation and normalization pipeline for training."""
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def get_val_transforms() -> transforms.Compose:
    """Evaluation / test normalization pipeline."""
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def count_images_in_dir(directory: str) -> int:
    """Counts image files with valid extensions in a directory."""
    if not os.path.exists(directory):
        return 0
    count = 0
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(VALID_EXTENSIONS):
                count += 1
    return count

def inspect_image_dataset(dataset_dir: str = "dataset") -> Dict[str, int]:
    """Returns the count of labeled images in undercut and no_undercut folders."""
    undercut_dir = os.path.join(dataset_dir, "undercut")
    no_undercut_dir = os.path.join(dataset_dir, "no_undercut")

    return {
        "undercut": count_images_in_dir(undercut_dir),
        "no_undercut": count_images_in_dir(no_undercut_dir),
        "total": count_images_in_dir(undercut_dir) + count_images_in_dir(no_undercut_dir)
    }

def get_dataloaders(
    dataset_dir: str = "dataset",
    batch_size: int = 16,
    val_split: float = 0.2,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, Dict[int, str]]:
    """
    Creates train and validation DataLoaders from dataset_dir.
    Expects subdirectories:
      dataset/no_undercut/
      dataset/undercut/
    """
    counts = inspect_image_dataset(dataset_dir)
    if counts["undercut"] == 0 or counts["no_undercut"] == 0:
        raise ValueError(
            f"Insufficient training images! Found {counts['undercut']} undercut images "
            f"and {counts['no_undercut']} normal images in '{dataset_dir}'. "
            f"Please populate 'dataset/undercut/' and 'dataset/no_undercut/' with gear photographs."
        )

    # Load dataset with ImageFolder
    full_dataset = datasets.ImageFolder(root=dataset_dir, transform=get_train_transforms())
    
    # Class mapping: ensures class names are properly matched
    class_to_idx = full_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, idx_to_class
