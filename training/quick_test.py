"""
PneumoScan — Quick Training Sanity Check
Location: training/quick_test.py

Runs 2 mini-epochs on a tiny subset of data to verify the full
training pipeline works end-to-end before committing hours to training.

Run this BEFORE starting full training to catch any issues early.
Takes about 2-3 minutes on CPU.

Usage:
    python training/quick_test.py
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models.lung.mobilenetv3  import build_mobilenetv3
from core.models.lung.densenet121  import build_densenet121
from core.models.lung.attention_cnn import build_attention_cnn
from training.dataset_loader import ChestXRayDataset
from training.augmentation   import get_train_transforms, get_val_transforms

DATA_DIR = 'domains/lung/data/processed'
DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'


def quick_test():
    print()
    print('=' * 55)
    print('  PneumoScan — Training Pipeline Sanity Check')
    print('=' * 55)
    print(f'  Device: {DEVICE}')
    print()

    # Load tiny subset — 64 images only
    try:
        train_ds = ChestXRayDataset(
            root_dir  = f'{DATA_DIR}/train',
            transform = get_train_transforms(224)
        )
        val_ds = ChestXRayDataset(
            root_dir  = f'{DATA_DIR}/val',
            transform = get_val_transforms(224)
        )
    except FileNotFoundError as e:
        print(f'  ERROR: {e}')
        print('  Make sure the dataset is in domains/lung/data/processed/')
        return False

    # Use only 64 training and 32 validation samples
    train_subset = Subset(train_ds, list(range(64)))
    val_subset   = Subset(val_ds,   list(range(32)))

    train_loader = DataLoader(train_subset, batch_size=8, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_subset,   batch_size=8, shuffle=False, num_workers=0)

    print(f'  Train subset: {len(train_subset)} images')
    print(f'  Val subset:   {len(val_subset)} images')
    print()

    # Test 3 representative models
    test_models = [
        ('MobileNetV3 (speed)',  build_mobilenetv3(pretrained=False)),
        ('DenseNet-121 (anchor)', build_densenet121(pretrained=False)),
        ('Attention CNN (custom)', build_attention_cnn()),
    ]

    criterion = nn.CrossEntropyLoss()
    all_passed = True

    for model_name, model in test_models:
        print(f'  Testing {model_name}...')
        model = model.to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        try:
            # 2 mini epochs
            for epoch in range(1, 3):
                # Train
                model.train()
                train_loss = 0.0
                for images, labels in train_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss    = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                # Validate
                model.eval()
                correct = 0
                total   = 0
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(DEVICE), labels.to(DEVICE)
                        outputs  = model(images)
                        preds    = outputs.argmax(dim=1)
                        correct += (preds == labels).sum().item()
                        total   += labels.size(0)

                val_acc = 100.0 * correct / total
                avg_loss = train_loss / len(train_loader)
                print(f'    Epoch {epoch}: loss={avg_loss:.4f}  val_acc={val_acc:.1f}%')

            print(f'  PASS  {model_name}')

        except Exception as e:
            print(f'  FAIL  {model_name}: {e}')
            all_passed = False

        print()

    print('=' * 55)
    if all_passed:
        print('  All checks passed.')
        print()
        print('  You are ready to start full training.')
        print()
        print('  Recommended training order (one at a time):')
        print('    python training/train.py --model densenet121')
        print('    python training/train.py --model resnet50')
        print('    python training/train.py --model efficientnet_b4')
        print('    python training/train.py --model mobilenetv3')
        print('    python training/train.py --model vit_b16')
        print('    python training/train.py --model inception_v3')
        print('    python training/train.py --model attention_cnn')
        print()
        print('  Or train all at once (takes many hours on CPU):')
        print('    python training/train.py --model all')
    else:
        print('  Some checks failed. Fix errors before training.')
    print('=' * 55)
    print()

    return all_passed


if __name__ == '__main__':
    quick_test()
