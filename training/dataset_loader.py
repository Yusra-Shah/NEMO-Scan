"""
NEMO Scan — Dataset Loader
Location: training/dataset_loader.py

PyTorch Dataset class and DataLoader factory for the lung module.
Reads images from the processed/ folder structure created by the Colab notebook.

Folder structure expected:
    domains/lung/data/processed/
        train/NORMAL/
        train/PNEUMONIA/
        val/NORMAL/
        val/PNEUMONIA/
        test/NORMAL/
        test/PNEUMONIA/
"""

import os
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A

# Import our augmentation pipelines
import sys
sys.path.append(str(Path(__file__).parent))
from augmentation import get_train_transforms, get_val_transforms


CLASS_TO_IDX = {"NORMAL": 0, "PNEUMONIA": 1}
IDX_TO_CLASS = {0: "NORMAL", 1: "PNEUMONIA"}


class ChestXRayDataset(Dataset):
    """
    PyTorch Dataset for chest X-ray images.

    Reads from a folder structured as:
        root/NORMAL/*.jpg
        root/PNEUMONIA/*.jpg

    Args:
        root_dir:   Path to the split folder (e.g. processed/train)
        transform:  Albumentations transform pipeline
        grayscale:  If True, convert images to 3-channel grayscale
                    (X-rays are grayscale but models expect 3 channels)
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[A.Compose] = None,
        grayscale: bool = True
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.grayscale = grayscale
        self.samples = []

        # Walk through NORMAL and PNEUMONIA subdirectories
        for class_name, class_idx in CLASS_TO_IDX.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Class folder not found: {class_dir}\n"
                    f"Run the Colab notebook first to prepare the dataset."
                )
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((img_path, class_idx))

        if len(self.samples) == 0:
            raise ValueError(
                f"No images found in {root_dir}. "
                f"Check that the dataset was downloaded and organized correctly."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path)

        # X-rays are grayscale — convert to RGB by repeating channels
        # All pretrained models expect 3-channel input
        if self.grayscale:
            image = image.convert('L')          # grayscale
            image = image.convert('RGB')        # back to 3-channel
        else:
            image = image.convert('RGB')

        image = np.array(image)

        # Apply augmentation / normalization transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label

    def get_class_counts(self) -> dict:
        """Returns count of images per class."""
        counts = {name: 0 for name in CLASS_TO_IDX}
        for _, label in self.samples:
            counts[IDX_TO_CLASS[label]] += 1
        return counts

    def get_class_weights(self) -> torch.Tensor:
        """
        Computes class weights for weighted loss.
        Used to handle class imbalance during training.
        Returns tensor of shape [num_classes].
        """
        counts = self.get_class_counts()
        total = sum(counts.values())
        weights = []
        for class_name in CLASS_TO_IDX:
            weight = total / (len(CLASS_TO_IDX) * counts[class_name])
            weights.append(weight)
        return torch.FloatTensor(weights)


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 2,
    image_size: int = 224,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates train, validation, and test DataLoaders.

    Args:
        data_dir:    Path to processed/ folder
        batch_size:  Images per batch (32 recommended for CPU)
        num_workers: Parallel data loading workers (2 for Windows)
        image_size:  Target image size (224 for all pretrained models)
        pin_memory:  Speed optimization for GPU (set False for CPU-only)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_dir = Path(data_dir)

    # Datasets
    train_dataset = ChestXRayDataset(
        root_dir=data_dir / 'train',
        transform=get_train_transforms(image_size)
    )
    val_dataset = ChestXRayDataset(
        root_dir=data_dir / 'val',
        transform=get_val_transforms(image_size)
    )
    test_dataset = ChestXRayDataset(
        root_dir=data_dir / 'test',
        transform=get_val_transforms(image_size)
    )

    # Windows requires num_workers=0 or 2 max to avoid multiprocessing issues
    safe_workers = min(num_workers, 2)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=safe_workers,
        pin_memory=pin_memory,
        drop_last=True      # drop incomplete last batch for stable batch norm
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=safe_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=safe_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader


def verify_dataset(data_dir: str) -> bool:
    """
    Verifies the dataset is correctly organized before training starts.
    Run this before training to catch setup issues early.

    Usage:
        python -c "from dataset_loader import verify_dataset; verify_dataset('domains/lung/data/processed')"
    """
    data_dir = Path(data_dir)
    print()
    print("=" * 50)
    print("  NEMO Scan — Dataset Verification")
    print("=" * 50)

    all_ok = True
    total_images = 0

    for split in ['train', 'val', 'test']:
        for cls in ['NORMAL', 'PNEUMONIA']:
            folder = data_dir / split / cls
            if not folder.exists():
                print(f"  MISSING  {split}/{cls}")
                all_ok = False
                continue
            count = len([f for f in folder.iterdir()
                         if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
            total_images += count
            status = "OK" if count > 0 else "EMPTY"
            print(f"  {status:7s}  {split:6s}/{cls:10s}  {count:,} images")

    print("-" * 50)
    print(f"  TOTAL: {total_images:,} images")
    print()

    if all_ok and total_images > 0:
        # Quick DataLoader test
        try:
            train_loader, val_loader, test_loader = get_dataloaders(
                data_dir, batch_size=4, num_workers=0
            )
            batch_imgs, batch_labels = next(iter(train_loader))
            print(f"  DataLoader test: batch shape {tuple(batch_imgs.shape)} — OK")
            print(f"  Labels in batch: {batch_labels.tolist()}")

            # Class weight check
            ds = train_loader.dataset
            weights = ds.get_class_weights()
            print(f"  Class weights: NORMAL={weights[0]:.3f}, PNEUMONIA={weights[1]:.3f}")
            print()
            print("  Dataset is ready for training.")
        except Exception as e:
            print(f"  DataLoader test FAILED: {e}")
            all_ok = False
    else:
        print("  Dataset not ready. Run the Colab notebook first.")

    print("=" * 50)
    print()
    return all_ok


if __name__ == "__main__":
    # Run verification when script is executed directly
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "domains/lung/data/processed"
    verify_dataset(data_path)
