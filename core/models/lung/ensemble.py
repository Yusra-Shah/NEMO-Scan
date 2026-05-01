"""
PneumoScan — Ensemble Voting System
Location: core/models/lung/ensemble.py

Manages all 7 models and produces the final weighted diagnosis.

Ensemble flow:
  1. MobileNetV3 runs immediately — instant result shown in GUI
  2. Remaining 5 voting models run in parallel background threads
  3. Each model outputs [P(Normal), P(Pneumonia)] via softmax
  4. Weighted average computed using validation-accuracy-based weights
  5. Final P(Pneumonia) > 0.50 threshold = Pneumonia diagnosis
  6. Severity determined from probability value
  7. AttentionCNN runs separately — produces heatmap only, not in vote

Output dictionary structure (matches GUI expectations exactly):
{
    "prediction":    "PNEUMONIA" | "NORMAL",
    "confidence":    float (0.0 - 1.0),
    "severity":      "None" | "Mild" | "Moderate" | "Severe",
    "subtype":       "Bacterial" | "Viral" | None,
    "model_votes":   { model_name: float, ... },
    "ensemble_prob": float,
    "heatmap_path":  str | None
}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import sys
from pathlib import Path

# Model imports
from .mobilenetv3  import MobileNetV3Lung,   build_mobilenetv3
from .resnet50     import ResNet50Lung,       build_resnet50
from .densenet121  import DenseNet121Lung,    build_densenet121
from .efficientnet import EfficientNetB4Lung, build_efficientnet_b4
from .vit          import ViTB16Lung,         build_vit_b16
from .inception    import InceptionV3Lung,    build_inception_v3
from .attention_cnn import AttentionCNN,      build_attention_cnn


# Ensemble weights — must sum to 1.0 across voting models
ENSEMBLE_WEIGHTS = {
    'mobilenetv3':    0.10,
    'resnet50':       0.18,
    'densenet121':    0.25,
    'efficientnet_b4':0.20,
    'vit_b16':        0.17,
    'inception_v3':   0.10,
    # attention_cnn is excluded from voting (weight = 0.00)
}

SEVERITY_THRESHOLDS = {
    'none':     (0.00, 0.30),
    'mild':     (0.30, 0.55),
    'moderate': (0.55, 0.75),
    'severe':   (0.75, 1.00),
}

CLASS_NAMES = ['NORMAL', 'PNEUMONIA']


def get_severity(pneumonia_prob: float) -> str:
    """Maps pneumonia probability to severity label."""
    for severity, (low, high) in SEVERITY_THRESHOLDS.items():
        if low <= pneumonia_prob < high:
            return severity.capitalize()
    return 'Severe'


def get_subtype(pneumonia_prob: float, model_votes: Dict) -> Optional[str]:
    """
    Heuristic subtype estimation.
    Higher confidence (>0.80) with strong consensus suggests Bacterial.
    Lower confidence or split votes suggests Viral.
    This is a secondary indicator — not a definitive diagnosis.
    """
    if pneumonia_prob < 0.50:
        return None

    pneumo_votes = [v for v in model_votes.values() if v > 0.50]
    avg_confidence = sum(pneumo_votes) / len(pneumo_votes) if pneumo_votes else 0.0

    if avg_confidence > 0.80:
        return 'Bacterial'
    else:
        return 'Viral'


class LungEnsemble(nn.Module):
    """
    Manages all 7 lung models and produces the final diagnosis.
    In production, models run in parallel threads from the inference engine.
    This class handles the voting logic and output formatting.
    """

    def __init__(self, device: str = 'cpu'):
        super().__init__()
        self.device = device

        # Voting models (6 models)
        self.voting_models = nn.ModuleDict({
            'mobilenetv3':     build_mobilenetv3(),
            'resnet50':        build_resnet50(),
            'densenet121':     build_densenet121(),
            'efficientnet_b4': build_efficientnet_b4(),
            'vit_b16':         build_vit_b16(),
            'inception_v3':    build_inception_v3(),
        })

        # Explainability model (separate, not in vote)
        self.attention_cnn = build_attention_cnn()

        self.to(device)

    def load_weights(self, weights_dir: str):
        """
        Loads trained .pth weight files for all models.
        Called by inference engine after training is complete.
        Skips models whose weight file does not exist yet.
        """
        weights_dir = Path(weights_dir)
        loaded = []
        missing = []

        for model_name, model in self.voting_models.items():
            weight_path = weights_dir / f'{model_name}.pth'
            if weight_path.exists():
                state = torch.load(weight_path, map_location=self.device)
                model.load_state_dict(state)
                model.eval()
                loaded.append(model_name)
            else:
                missing.append(model_name)

        # Attention CNN
        attn_path = weights_dir / 'attention_cnn.pth'
        if attn_path.exists():
            state = torch.load(attn_path, map_location=self.device)
            self.attention_cnn.load_state_dict(state)
            self.attention_cnn.eval()
            loaded.append('attention_cnn')
        else:
            missing.append('attention_cnn')

        print(f'Loaded weights: {loaded}')
        if missing:
            print(f'Missing weights (not yet trained): {missing}')

    def predict_single(
        self,
        model_name: str,
        image_tensor: torch.Tensor
    ) -> float:
        """
        Runs a single model and returns P(Pneumonia).
        Called from parallel threads in the inference engine.
        """
        model = self.voting_models[model_name]
        model.eval()

        with torch.no_grad():
            logits = model(image_tensor.to(self.device))
            probs = F.softmax(logits, dim=1)
            pneumonia_prob = probs[0, 1].item()

        return pneumonia_prob

    def predict_ensemble(
        self,
        image_tensor: torch.Tensor,
        heatmap_path: Optional[str] = None
    ) -> Dict:
        """
        Runs all 6 voting models and produces the final diagnosis.
        In the real GUI flow, MobileNetV3 runs first and this is called
        after all models complete. Returns the full output dictionary.

        Args:
            image_tensor: Preprocessed image tensor (1, 3, 224, 224)
            heatmap_path: Path to saved Grad-CAM image (if generated)

        Returns:
            Complete diagnosis dictionary matching GUI expectations
        """
        model_votes = {}

        for model_name, model in self.voting_models.items():
            pneumonia_prob = self.predict_single(model_name, image_tensor)
            model_votes[model_name] = pneumonia_prob

        # Weighted average across voting models
        ensemble_prob = sum(
            model_votes[name] * ENSEMBLE_WEIGHTS[name]
            for name in model_votes
        )

        # Final decision
        prediction = 'PNEUMONIA' if ensemble_prob >= 0.50 else 'NORMAL'
        severity   = get_severity(ensemble_prob)
        subtype    = get_subtype(ensemble_prob, model_votes)

        # Confidence = distance from decision boundary, scaled to 0-1
        if prediction == 'PNEUMONIA':
            confidence = ensemble_prob
        else:
            confidence = 1.0 - ensemble_prob

        return {
            'prediction':    prediction,
            'confidence':    round(confidence, 4),
            'severity':      severity,
            'subtype':       subtype,
            'model_votes':   {k: round(v, 4) for k, v in model_votes.items()},
            'ensemble_prob': round(ensemble_prob, 4),
            'heatmap_path':  heatmap_path
        }

    def set_eval_mode(self):
        """Set all models to evaluation mode."""
        for model in self.voting_models.values():
            model.eval()
        self.attention_cnn.eval()

    def freeze_all_backbones(self):
        """Freeze all model backbones. Used in Phase 1 training."""
        for model in self.voting_models.values():
            model.freeze_backbone()
        self.attention_cnn.freeze_backbone()

    def unfreeze_all_backbones(self):
        """Unfreeze all backbones for Phase 2 fine-tuning."""
        for model in self.voting_models.values():
            model.unfreeze_backbone()
        self.attention_cnn.unfreeze_backbone()


def verify_models():
    """
    Instantiates all 7 models and runs a dummy forward pass.
    Run this to confirm all models are correctly defined before training.

    Usage:
        python -c "from ensemble import verify_models; verify_models()"
    """
    print()
    print('=' * 55)
    print('  PneumoScan — Model Verification')
    print('=' * 55)

    dummy = torch.zeros(2, 3, 224, 224)  # batch of 2

    builders = [
        ('MobileNetV3',    build_mobilenetv3,    False),
        ('ResNet-50',      build_resnet50,        False),
        ('DenseNet-121',   build_densenet121,     False),
        ('EfficientNet-B4',build_efficientnet_b4, False),
        ('ViT-B/16',       build_vit_b16,         False),
        ('InceptionV3',    build_inception_v3,    False),
        ('Attention CNN',  build_attention_cnn,   True),   # no pretrained arg
    ]

    total_params = 0
    all_passed = True

    for name, builder, no_pretrained in builders:
        try:
            if no_pretrained:
                model = builder()
            else:
                model = builder(pretrained=False)  # skip download during verify

            model.eval()
            with torch.no_grad():
                out = model(dummy)

            params = sum(p.numel() for p in model.parameters()) / 1e6
            total_params += params
            assert out.shape == (2, 2), f"Expected (2,2), got {out.shape}"
            print(f'  PASS  {name:<20} output: {tuple(out.shape)}   params: {params:.1f}M')

        except Exception as e:
            print(f'  FAIL  {name:<20} ERROR: {e}')
            all_passed = False

    # Verify ensemble weights sum to 1.0
    weight_sum = sum(ENSEMBLE_WEIGHTS.values())
    weight_ok = abs(weight_sum - 1.0) < 0.001
    print()
    print(f'  Ensemble weights sum: {weight_sum:.3f}  {"OK" if weight_ok else "ERROR - must sum to 1.0"}')
    print(f'  Total parameters: {total_params:.1f}M')
    print()

    if all_passed and weight_ok:
        print('  All models verified. Ready for training.')
    else:
        print('  Some models failed. Check errors above.')
    print('=' * 55)
    print()


if __name__ == '__main__':
    verify_models()
