"""
PneumoScan — Evaluation Script
Location: training/evaluate.py

Evaluates a trained model on the test set and produces:
  - Accuracy, Precision, Recall, F1 Score
  - AUC-ROC score
  - Confusion matrix
  - Per-class performance breakdown

Usage:
    python training/evaluate.py --model densenet121
    python training/evaluate.py --model all
"""

import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models.lung.mobilenetv3   import build_mobilenetv3
from core.models.lung.resnet50      import build_resnet50
from core.models.lung.densenet121   import build_densenet121
from core.models.lung.efficientnet  import build_efficientnet_b4
from core.models.lung.vit           import build_vit_b16
from core.models.lung.inception     import build_inception_v3
from core.models.lung.attention_cnn import build_attention_cnn
from training.dataset_loader        import get_dataloaders

DATA_DIR    = 'domains/lung/data/processed'
WEIGHTS_DIR = 'weights/lung'
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_BUILDERS = {
    'mobilenetv3':     lambda: build_mobilenetv3(pretrained=False),
    'resnet50':        lambda: build_resnet50(pretrained=False),
    'densenet121':     lambda: build_densenet121(pretrained=False),
    'efficientnet_b4': lambda: build_efficientnet_b4(pretrained=False),
    'vit_b16':         lambda: build_vit_b16(pretrained=False),
    'inception_v3':    lambda: build_inception_v3(pretrained=False),
    'attention_cnn':   lambda: build_attention_cnn(),
}


def compute_metrics(all_labels, all_preds, all_probs):
    """Computes classification metrics without sklearn dependency issues."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix
    )

    accuracy  = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, zero_division=0) * 100
    recall    = recall_score(all_labels, all_preds, zero_division=0) * 100
    f1        = f1_score(all_labels, all_preds, zero_division=0) * 100
    auc       = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    cm        = confusion_matrix(all_labels, all_preds)

    return {
        'accuracy':  accuracy,
        'precision': precision,
        'recall':    recall,
        'f1':        f1,
        'auc':       auc,
        'confusion_matrix': cm
    }


def evaluate_model(model_name: str):
    """Loads trained weights and evaluates on test set."""

    weight_path = Path(WEIGHTS_DIR) / f'{model_name}.pth'
    if not weight_path.exists():
        print(f'  Weight file not found: {weight_path}')
        print(f'  Train this model first: python training/train.py --model {model_name}')
        return None

    print()
    print('=' * 60)
    print(f'  Evaluating: {model_name.upper()}')
    print('=' * 60)

    # Load model
    model = MODEL_BUILDERS[model_name]()
    state = torch.load(weight_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    model.to(DEVICE)

    # Test loader
    _, _, test_loader = get_dataloaders(
        data_dir=DATA_DIR, batch_size=32, num_workers=2
    )

    all_labels = []
    all_preds  = []
    all_probs  = []

    with torch.no_grad():
        for images, labels in test_loader:
            images  = images.to(DEVICE)
            outputs = model(images)
            probs   = F.softmax(outputs, dim=1)

            preds = outputs.argmax(dim=1).cpu().numpy()
            pneumo_probs = probs[:, 1].cpu().numpy()

            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(pneumo_probs)

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    cm = metrics['confusion_matrix']

    # Print results
    print(f'  Accuracy:  {metrics["accuracy"]:.2f}%')
    print(f'  Precision: {metrics["precision"]:.2f}%')
    print(f'  Recall:    {metrics["recall"]:.2f}%')
    print(f'  F1 Score:  {metrics["f1"]:.2f}%')
    print(f'  AUC-ROC:   {metrics["auc"]:.4f}')
    print()
    print('  Confusion Matrix:')
    print(f'                Predicted')
    print(f'                NORMAL   PNEUMONIA')
    print(f'  Actual NORMAL    {cm[0][0]:4d}      {cm[0][1]:4d}')
    print(f'  Actual PNEUMO    {cm[1][0]:4d}      {cm[1][1]:4d}')
    print()

    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    print(f'  Sensitivity (Recall):  {sensitivity:.2f}%')
    print(f'  Specificity:           {specificity:.2f}%')
    print()

    # Medical AI note
    print('  Note: In medical AI, Recall matters more than Precision.')
    print('  Missing a real pneumonia case (False Negative) is more')
    print('  dangerous than a false alarm (False Positive).')
    print('  Target: Recall > 90%, AUC > 0.95')
    print('=' * 60)

    return metrics


def main():
    parser = argparse.ArgumentParser(description='PneumoScan — Model Evaluation')
    parser.add_argument(
        '--model',
        type=str,
        default='densenet121',
        choices=list(MODEL_BUILDERS.keys()) + ['all']
    )
    args = parser.parse_args()

    if args.model == 'all':
        all_results = {}
        for model_name in MODEL_BUILDERS:
            result = evaluate_model(model_name)
            if result:
                all_results[model_name] = result

        if all_results:
            print()
            print('=' * 60)
            print('  ENSEMBLE EVALUATION SUMMARY')
            print('=' * 60)
            print(f'  {"Model":<20} {"Acc":>7} {"Recall":>8} {"AUC":>7}')
            print('-' * 60)
            for name, m in all_results.items():
                print(f'  {name:<20} {m["accuracy"]:>6.2f}% '
                      f'{m["recall"]:>7.2f}% '
                      f'{m["auc"]:>7.4f}')
            print('=' * 60)
    else:
        evaluate_model(args.model)


if __name__ == '__main__':
    main()
