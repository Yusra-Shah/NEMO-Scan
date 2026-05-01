"""
PneumoScan — Model 3: DenseNet-121
Location: core/models/lung/densenet121.py

Role: Anchor model. Highest ensemble weight. Most important model.
This is the SAME architecture used in Stanford's CheXNet (2017),
which matched radiologist-level performance on chest X-ray diagnosis.

Architecture: Dense connections — every layer receives feature maps
from ALL preceding layers. This means gradients flow directly to
early layers, solving vanishing gradients more aggressively than
ResNet skip connections. Particularly powerful for subtle medical
image patterns.

Expected accuracy: 93-94% (highest of all 7 models)
Ensemble weight: 0.25
"""

import torch
import torch.nn as nn
import timm


class DenseNet121Lung(nn.Module):
    """
    DenseNet-121 fine-tuned for pneumonia detection.
    The CheXNet architecture. Receives highest ensemble voting weight.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()

        self.backbone = timm.create_model(
            'densenet121',
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )

        feature_dim = self.backbone.num_features  # 1024 for DenseNet-121

        # Slightly deeper head to match CheXNet design philosophy
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

        self.model_name = 'densenet121'
        self.ensemble_weight = 0.25

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
        """
        DenseNet gets more conservative fine-tuning LR.
        Its dense connections make it sensitive to large weight updates.
        """
        return [
            {'params': self.backbone.parameters(), 'lr': base_lr * 0.05},
            {'params': self.classifier.parameters(), 'lr': base_lr}
        ]


def build_densenet121(num_classes: int = 2, pretrained: bool = True) -> DenseNet121Lung:
    return DenseNet121Lung(num_classes=num_classes, pretrained=pretrained)
