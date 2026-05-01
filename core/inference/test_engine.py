"""
PneumoScan - Inference Engine Test
Location: core/inference/test_engine.py

Run this to verify the inference engine loads all models correctly
and can process a dummy image.

Usage:
    cd D:\PneumoScan\Pneumo
    python core/inference/test_engine.py
"""

import sys
import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.inference.engine import InferenceEngine


def create_dummy_xray(path: str):
    """Creates a dummy grayscale X-ray image for testing."""
    img = Image.fromarray(
        (np.random.rand(512, 512) * 255).astype(np.uint8), mode='L'
    ).convert('RGB')
    img.save(path)
    print(f'Created dummy X-ray: {path}')


def main():
    print()
    print('=' * 55)
    print('  PneumoScan - Inference Engine Test')
    print('=' * 55)

    # Initialize engine
    engine = InferenceEngine()

    # Load all models
    weights_dir = ROOT / 'weights' / 'lung'
    status = engine.load_models(str(weights_dir))

    print()
    print('Model load status:')
    for name, loaded in status.items():
        icon = 'OK  ' if loaded else 'MISS'
        print(f'  {icon}  {name}')

    loaded_count = sum(status.values())
    if loaded_count == 0:
        print()
        print('No models loaded. Make sure weight files are in weights/lung/')
        return

    # Create a dummy test image
    test_image = str(ROOT / 'outputs' / 'test_dummy.jpg')
    os.makedirs(ROOT / 'outputs', exist_ok=True)
    create_dummy_xray(test_image)

    print()
    print('Testing quick prediction (MobileNetV3)...')
    quick = engine.predict_quick(test_image)
    print(f'  Result: {quick}')

    print()
    print('Testing full ensemble prediction...')

    def on_model_done(model_name, prob):
        print(f'  {model_name}: P(Pneumonia) = {prob:.4f}')

    result = engine.predict_full(
        test_image,
        heatmaps_dir=str(ROOT / 'outputs' / 'heatmaps'),
        progress_callback=on_model_done,
    )

    print()
    print('Full result:')
    for key, val in result.items():
        if key != 'model_votes':
            print(f'  {key}: {val}')

    print()
    print('Model votes:')
    for model, vote in result.get('model_votes', {}).items():
        print(f'  {model}: {vote}')

    print()
    print('Inference engine is working correctly.')
    print('=' * 55)

    # Clean up dummy image
    if os.path.exists(test_image):
        os.remove(test_image)


if __name__ == '__main__':
    main()
