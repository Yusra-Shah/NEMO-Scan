"""
NEMO Scan — Augmentation Pipeline
Location: training/augmentation.py

Defines all augmentation transforms used during:
  1. Dataset preparation (Colab) — to generate synthetic images
  2. Training — applied live to each batch
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size: int = 224) -> A.Compose:
    """
    Training augmentation pipeline.
    Applied live to every batch during model training.
    Simulates real-world X-ray variation across hospitals and equipment.
    """
    return A.Compose([
        # Geometric transforms
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=12, p=0.7),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.05,
            rotate_limit=0,
            p=0.4
        ),

        # X-ray specific contrast enhancement
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),

        # Exposure simulation (different X-ray machines)
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.6
        ),

        # Sensor noise simulation
        A.GaussNoise(p=0.4),

        # Soft tissue / breathing artifact simulation
        A.ElasticTransform(
            alpha=1,
            sigma=50,
            p=0.3
        ),

        # Geometric distortion (older equipment)
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.3,
            p=0.3
        ),

        # Standardize size
        A.Resize(image_size, image_size),

        # Normalize to ImageNet mean/std (required for pretrained models)
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),

        ToTensorV2()
    ])


def get_val_transforms(image_size: int = 224) -> A.Compose:
    """
    Validation and test transforms.
    No augmentation — only resize and normalize.
    Must be identical to what models expect at inference time.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_inference_transforms(image_size: int = 224) -> A.Compose:
    """
    Inference transforms — used by the inference engine when
    a doctor uploads an X-ray through the GUI.
    Applies CLAHE first to enhance contrast before normalization.
    """
    return A.Compose([
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_colab_augmentation_pipeline(image_size: int = 224) -> A.Compose:
    """
    Augmentation pipeline used in Colab during dataset preparation.
    Slightly heavier than training augmentation since these become
    permanent synthetic images in the dataset.
    Does NOT include normalization or ToTensorV2 — outputs raw numpy images.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=12, p=0.7),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.6
        ),
        A.GaussNoise(p=0.4),
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
        A.Resize(image_size, image_size),
    ])
