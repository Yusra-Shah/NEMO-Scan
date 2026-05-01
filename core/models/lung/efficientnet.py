"""
PneumoScan — Model 4: EfficientNet-B4
Location: core/models/lung/efficientnet.py

Role: Accuracy-efficiency balance. Compound scaling model.
Google's mathematically optimal scaling formula applied simultaneously
to network width, depth, and input resolution. Achieves ResNet-level
accuracy with significantly fewer parameters.

Particularly good at fine-grained pattern recognition — useful for
detecting subtle pneumonia opacity patterns that other models miss.

Expected accuracy: 92-94%
Ensemble weight: 0.20
"""

import torch
import torch.nn as nn
import timm


class EfficientNetB4Lung(nn.Module):
    """
    EfficientNet-B4 fine-tuned for pneumonia detection.
    B4 variant chosen as the best balance of accuracy and CPU speed.
    B0-B3 are faster but less accurate. B5+ are too slow for CPU inference.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()

        self.backbone = timm.create_model(
            'efficientnet_b4',
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )

        feature_dim = self.backbone.num_features  # 1792 for EfficientNet-B4

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(feature_dim, 512),
            nn.SiLU(),              # SiLU (Swish) matches EfficientNet's internal activations
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )

        self.model_name = 'efficientnet_b4'
        self.ensemble_weight = 0.20

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_parameter_groups(self, base_lr: float = 0.001):
        return [
            {'params': self.backbone.parameters(), 'lr': base_lr * 0.1},
            {'params': self.classifier.parameters(), 'lr': base_lr}
        ]


def build_efficientnet_b4(num_classes: int = 2, pretrained: bool = True) -> EfficientNetB4Lung:
    return EfficientNetB4Lung(num_classes=num_classes, pretrained=pretrained)
