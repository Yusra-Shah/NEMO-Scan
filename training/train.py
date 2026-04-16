"""
NEMO Scan — Training Script
Location: training/train.py

Trains all 7 lung models using transfer learning.
Two-phase training strategy:
  Phase 1: Freeze backbone, train classifier head only (fast, 5 epochs)
  Phase 2: Unfreeze backbone, fine-tune entire network (slower, up to 25 epochs)

Usage:
    # Train a single model
    python training/train.py --model densenet121

    # Train all 7 models sequentially
    python training/train.py --model all

    # Train with custom settings
    python training/train.py --model resnet50 --epochs 20 --batch_size 16
"""

import sys
import os
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models.lung.mobilenetv3   import build_mobilenetv3
from core.models.lung.resnet50      import build_resnet50
from core.models.lung.densenet121   import build_densenet121
from core.models.lung.efficientnet  import build_efficientnet_b4
from core.models.lung.vit           import build_vit_b16
from core.models.lung.inception     import build_inception_v3
from core.models.lung.attention_cnn import build_attention_cnn

from training.dataset_loader import get_dataloaders

# ── Configuration ────────────────────────────────────────────────────────────

DATA_DIR        = 'domains/lung/data/processed'
WEIGHTS_DIR     = 'weights/lung'
CHECKPOINT_DIR  = 'domains/lung/checkpoints'

PHASE1_EPOCHS   = 5       # freeze backbone, train head only
PHASE2_EPOCHS   = 25      # full fine-tuning
BATCH_SIZE      = 32
NUM_WORKERS     = 2
IMAGE_SIZE      = 224
BASE_LR         = 0.001
PATIENCE        = 7       # early stopping patience

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_BUILDERS = {
    'mobilenetv3':     lambda: build_mobilenetv3(pretrained=True),
    'resnet50':        lambda: build_resnet50(pretrained=True),
    'densenet121':     lambda: build_densenet121(pretrained=True),
    'efficientnet_b4': lambda: build_efficientnet_b4(pretrained=True),
    'vit_b16':         lambda: build_vit_b16(pretrained=True),
    'inception_v3':    lambda: build_inception_v3(pretrained=True),
    'attention_cnn':   lambda: build_attention_cnn(),  # no pretrained
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        return torch.device('cuda')
    print('GPU not available — training on CPU.')
    print('Expected time per model: 2-4 hours on CPU.')
    return torch.device('cpu')


def save_checkpoint(model, path: str, epoch: int, val_acc: float):
    """Saves model state dict as a checkpoint."""
    torch.save({
        'epoch':     epoch,
        'val_acc':   val_acc,
        'state_dict': model.state_dict()
    }, path)


def load_best_weights(model, checkpoint_path: str) -> float:
    """Loads best checkpoint weights into model. Returns best val_acc."""
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    return checkpoint['val_acc']


class EarlyStopping:
    """Stops training when validation accuracy stops improving."""

    def __init__(self, patience: int = 7, min_delta: float = 0.001):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_acc   = 0.0
        self.should_stop = False

    def __call__(self, val_acc: float) -> bool:
        if val_acc > self.best_acc + self.min_delta:
            self.best_acc = val_acc
            self.counter  = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ── Training Loop ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    optimizer:  optim.Optimizer,
    criterion:  nn.Module,
    device:     torch.device,
    epoch:      int,
    total_epochs: int
) -> tuple:
    """Runs one full training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping — prevents exploding gradients (important for ViT)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        predicted   = outputs.argmax(dim=1)
        correct    += (predicted == labels).sum().item()
        total      += labels.size(0)

        # Progress print every 50 batches
        if (batch_idx + 1) % 50 == 0:
            running_acc = 100.0 * correct / total
            print(f'    Epoch [{epoch}/{total_epochs}] '
                  f'Batch [{batch_idx+1}/{len(loader)}] '
                  f'Loss: {loss.item():.4f}  '
                  f'Acc: {running_acc:.2f}%')

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate(
    model:    nn.Module,
    loader:   DataLoader,
    criterion: nn.Module,
    device:   torch.device
) -> tuple:
    """Evaluates model on val/test set. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.no_grad():
        for images, labels in loader:
            images  = images.to(device)
            labels  = labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss += loss.item()
            predicted   = outputs.argmax(dim=1)
            correct    += (predicted == labels).sum().item()
            total      += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ── Main Training Function ────────────────────────────────────────────────────

def train_model(model_name: str, device: torch.device):
    """
    Full training pipeline for a single model.
    Phase 1: freeze backbone, train head (5 epochs)
    Phase 2: unfreeze, fine-tune all layers (up to 25 epochs)
    """
    print()
    print('=' * 60)
    print(f'  Training: {model_name.upper()}')
    print(f'  Device:   {device}')
    print('=' * 60)

    # Paths
    Path(WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    checkpoint_path = f'{CHECKPOINT_DIR}/{model_name}_best.pth'
    final_path      = f'{WEIGHTS_DIR}/{model_name}.pth'

    # Data
    train_loader, val_loader, _ = get_dataloaders(
        data_dir    = DATA_DIR,
        batch_size  = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        image_size  = IMAGE_SIZE
    )

    # Class weights for imbalanced loss (balanced dataset = equal weights)
    class_weights = train_loader.dataset.get_class_weights().to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    # Build model
    model = MODEL_BUILDERS[model_name]().to(device)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss':   [], 'val_acc':   []
    }

    best_val_acc  = 0.0
    start_time    = time.time()

    # ── Phase 1: Train classifier head only ──────────────────────────────────
    print()
    print(f'  Phase 1: Classifier head training ({PHASE1_EPOCHS} epochs)')
    print(f'  Backbone frozen — only top layers update')
    print()

    if hasattr(model, 'freeze_backbone'):
        model.freeze_backbone()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=BASE_LR
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=PHASE1_EPOCHS, eta_min=1e-5)

    for epoch in range(1, PHASE1_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, PHASE1_EPOCHS
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'  [P1 Epoch {epoch}/{PHASE1_EPOCHS}] '
              f'Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  '
              f'Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, checkpoint_path, epoch, val_acc)
            print(f'  Checkpoint saved — Val Acc: {val_acc:.2f}%')

    # ── Phase 2: Fine-tune entire network ────────────────────────────────────
    print()
    print(f'  Phase 2: Full fine-tuning (up to {PHASE2_EPOCHS} epochs)')
    print(f'  Backbone unfrozen — all layers update at reduced LR')
    print()

    if hasattr(model, 'unfreeze_backbone'):
        model.unfreeze_backbone()

    # Use differential learning rates if model supports it
    if hasattr(model, 'get_parameter_groups'):
        param_groups = model.get_parameter_groups(base_lr=BASE_LR)
    else:
        param_groups = [{'params': model.parameters(), 'lr': BASE_LR * 0.1}]

    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=PHASE2_EPOCHS, eta_min=1e-6)
    early_stopper = EarlyStopping(patience=PATIENCE)

    for epoch in range(1, PHASE2_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch, PHASE2_EPOCHS
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'  [P2 Epoch {epoch}/{PHASE2_EPOCHS}] '
              f'Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  '
              f'Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, checkpoint_path, epoch, val_acc)
            print(f'  Checkpoint saved — Best Val Acc: {val_acc:.2f}%')

        if early_stopper(val_acc):
            print(f'  Early stopping at epoch {epoch}. '
                  f'No improvement for {PATIENCE} epochs.')
            break

    # ── Save final weights ────────────────────────────────────────────────────
    # Load best checkpoint weights before saving final
    best_acc = load_best_weights(model, checkpoint_path)
    torch.save(model.state_dict(), final_path)

    elapsed = (time.time() - start_time) / 60
    print()
    print('=' * 60)
    print(f'  {model_name.upper()} Training Complete')
    print(f'  Best Val Accuracy: {best_acc:.2f}%')
    print(f'  Time: {elapsed:.1f} minutes')
    print(f'  Weights saved: {final_path}')
    print('=' * 60)
    print()

    return best_acc, history


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='NEMO Scan — Model Training')
    parser.add_argument(
        '--model',
        type=str,
        default='densenet121',
        choices=list(MODEL_BUILDERS.keys()) + ['all'],
        help='Model to train. Use "all" to train all 7 sequentially.'
    )
    parser.add_argument('--epochs',     type=int, default=PHASE2_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr',         type=float, default=BASE_LR)
    args = parser.parse_args()

    device = get_device()

    if args.model == 'all':
        # Recommended training order:
        # Start with DenseNet (most important), then others
        order = [
            'densenet121',
            'resnet50',
            'efficientnet_b4',
            'mobilenetv3',
            'vit_b16',
            'inception_v3',
            'attention_cnn'
        ]
        print()
        print('Training all 7 models sequentially.')
        print(f'Order: {" -> ".join(order)}')
        print()

        results = {}
        for model_name in order:
            acc, _ = train_model(model_name, device)
            results[model_name] = acc

        print()
        print('=' * 60)
        print('  ALL MODELS TRAINING COMPLETE')
        print('=' * 60)
        for name, acc in results.items():
            print(f'  {name:<20} Best Val Acc: {acc:.2f}%')
        print('=' * 60)

    else:
        train_model(args.model, device)


if __name__ == '__main__':
    main()
