"""
NEMO Scan — Image Preprocessing Utilities
Location: utils/preprocessing.py

Handles all image preprocessing for:
  - Inference (when a doctor uploads an X-ray via GUI)
  - Preprocessing preview shown in the scan panel
  - CLAHE contrast enhancement
  - DICOM support placeholder (future)
"""

from pathlib import Path
from typing import Tuple, Optional, Union
import numpy as np
import cv2
from PIL import Image
import torch


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Loads an image from disk and converts to RGB numpy array.
    Handles JPEG, PNG. DICOM support is a future placeholder.

    Args:
        image_path: Path to the image file

    Returns:
        RGB numpy array of shape (H, W, 3)
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    suffix = image_path.suffix.lower()

    if suffix in ['.jpg', '.jpeg', '.png']:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    elif suffix == '.dcm':
        # DICOM placeholder — will be implemented in future version
        raise NotImplementedError(
            "DICOM support is planned for a future version. "
            "Please convert to JPEG or PNG first."
        )
    else:
        raise ValueError(f"Unsupported format: {suffix}. Use JPEG or PNG.")

    return img


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Enhances local contrast in X-ray images to reveal subtle features.
    Applied per channel on RGB input.

    Args:
        image:          RGB numpy array
        clip_limit:     Threshold for contrast limiting (2.0 recommended)
        tile_grid_size: Size of grid for histogram equalization

    Returns:
        CLAHE-enhanced RGB numpy array
    """
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )
    # Apply CLAHE to each channel independently
    channels = cv2.split(image)
    enhanced = [clahe.apply(ch) for ch in channels]
    return cv2.merge(enhanced)


def resize_image(
    image: np.ndarray,
    target_size: int = 224
) -> np.ndarray:
    """
    Resizes image to target_size x target_size.
    Uses LANCZOS interpolation for downscaling (best quality).

    Args:
        image:       Input numpy array
        target_size: Target width and height in pixels

    Returns:
        Resized numpy array
    """
    return cv2.resize(
        image,
        (target_size, target_size),
        interpolation=cv2.INTER_LANCZOS4
    )


def normalize_imagenet(image: np.ndarray) -> np.ndarray:
    """
    Normalizes image using ImageNet mean and std.
    Required for all pretrained models (ResNet, DenseNet, ViT, etc.)

    Args:
        image: RGB numpy array with values in [0, 255]

    Returns:
        Normalized float32 numpy array
    """
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    return image


def preprocess_for_inference(
    image_path: Union[str, Path],
    image_size: int = 224,
    apply_clahe_enhancement: bool = True
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Full preprocessing pipeline for inference.
    Used by the inference engine when a doctor uploads an X-ray.

    Args:
        image_path:              Path to uploaded X-ray image
        image_size:              Target size (224 for all models)
        apply_clahe_enhancement: Whether to apply CLAHE before inference

    Returns:
        Tuple of:
          - tensor: Float32 tensor of shape (1, 3, H, W) ready for model input
          - preview: RGB numpy array for GUI heatmap overlay (before normalization)
    """
    # Load
    image = load_image(image_path)

    # Convert grayscale X-rays to 3-channel
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # CLAHE enhancement
    if apply_clahe_enhancement:
        image = apply_clahe(image)

    # Resize
    image_resized = resize_image(image, image_size)

    # Save a copy before normalization — used for heatmap overlay in GUI
    preview = image_resized.copy()

    # Normalize
    image_normalized = normalize_imagenet(image_resized)

    # Convert to tensor: (H, W, C) -> (C, H, W) -> (1, C, H, W)
    tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.float()

    return tensor, preview


def preprocess_for_preview(
    image_path: Union[str, Path],
    target_size: int = 400
) -> np.ndarray:
    """
    Lightweight preprocessing for GUI preview display only.
    Does NOT normalize — returns displayable RGB image.
    Used to show the uploaded X-ray in the scan panel before analysis.

    Args:
        image_path:  Path to uploaded image
        target_size: Display size for GUI preview

    Returns:
        RGB numpy array suitable for display
    """
    image = load_image(image_path)

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Apply CLAHE for better visualization
    image = apply_clahe(image)

    # Resize for display (larger than model input)
    image = cv2.resize(image, (target_size, target_size),
                       interpolation=cv2.INTER_LANCZOS4)
    return image


def numpy_to_qimage(image: np.ndarray):
    """
    Converts a numpy RGB array to a QImage for display in PySide6.
    Used by the GUI scan panel to display the uploaded X-ray.

    Args:
        image: RGB numpy array of shape (H, W, 3)

    Returns:
        PySide6 QImage object
    """
    from PySide6.QtGui import QImage

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    height, width, channels = image.shape
    bytes_per_line = channels * width
    # QImage expects RGB888 format
    q_image = QImage(
        image.tobytes(),
        width,
        height,
        bytes_per_line,
        QImage.Format.Format_RGB888
    )
    return q_image
