"""
PneumoScan — Model 6: InceptionV3
Location: core/models/lung/inception.py

Role: Multi-scale detector. Catches both fine texture details and
coarse structural patterns simultaneously in a single forward pass.

Architecture: Inception modules apply multiple filter sizes (1x1, 3x3,
5x5) in parallel at each layer and concatenate the results. This means
the model detects fine-grained local texture (early consolidation,
ground-glass opacity) AND large structural patterns (lobar consolidation,
pleural effusion) at the same time.

Note: InceptionV3 requires 299x299 input. We resize internally.
Expected accuracy: 91-92%
Ensemble weight: 0.10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class InceptionV3Lung(nn.Module):
    """
    InceptionV3 fine-tuned for pneumonia detection.
    Handles the 299x299 input requirement internally via interpolation.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()

        # InceptionV3 needs 299x299 — timm handles this internally
        self.backbone = timm.create_model(
            'inception_v3',
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )

        feature_dim = self.backbone.num_features  # 2048 for InceptionV3

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )

        self.model_name = 'inception_v3'
        self.ensemble_weight = 0.10
        self.input_size = 299   # InceptionV3 native input size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Resize from 224 to 299 if needed
        if x.shape[-1] != self.input_size:
            x = F.interpolate(
                x,
                size=(self.input_size, self.input_size),
                mode='bilinear',
                align_corners=False
            )
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


def build_inception_v3(num_classes: int = 2, pretrained: bool = True) -> InceptionV3Lung:
    return InceptionV3Lung(num_classes=num_classes, pretrained=pretrained)
