"""
PneumoScan — Model 2: ResNet-50
Location: core/models/lung/resnet50.py

Role: Core ensemble backbone. Strong general-purpose feature extractor.
Introduced skip connections (residual shortcuts) that solved the
vanishing gradient problem, allowing very deep networks to train.

Architecture: 50 layers with skip connections between layer groups.
Each layer learns progressively more abstract features.
Expected accuracy: 92-93%
Ensemble weight: 0.18
"""

import torch
import torch.nn as nn
import timm


class ResNet50Lung(nn.Module):
    """
    ResNet-50 fine-tuned for pneumonia detection.
    Pretrained on ImageNet, final FC layer replaced for binary output.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()

        self.backbone = timm.create_model(
            'resnet50',
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )

        feature_dim = self.backbone.num_features  # 2048 for ResNet-50

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        self.model_name = 'resnet50'
        self.ensemble_weight = 0.18

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


def build_resnet50(num_classes: int = 2, pretrained: bool = True) -> ResNet50Lung:
    return ResNet50Lung(num_classes=num_classes, pretrained=pretrained)
