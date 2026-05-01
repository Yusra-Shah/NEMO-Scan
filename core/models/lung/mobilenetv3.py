"""
PneumoScan — Model 1: MobileNetV3
Location: core/models/lung/mobilenetv3.py

Role: Speed model. Runs first, returns result in 2-3 seconds on CPU.
The user sees this result immediately while the other 6 models
continue running in background threads.

Architecture: Google's depthwise separable convolutions.
Achieves 90% of ResNet accuracy at 10% of the compute cost.
Expected accuracy: 89-91%
Ensemble weight: 0.10
"""

import torch
import torch.nn as nn
import timm


class MobileNetV3Lung(nn.Module):
    """
    MobileNetV3-Large fine-tuned for pneumonia detection.
    Pretrained on ImageNet, final classifier replaced for binary output.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()

        # Load pretrained backbone from timm
        self.backbone = timm.create_model(
            'mobilenetv3_large_100',
            pretrained=pretrained,
            num_classes=0,      # remove original classifier
            global_pool='avg'
        )

        # Get feature dimension from backbone
        feature_dim = 1280  # actual output dim for mobilenetv3_large_100

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

        self.model_name = 'mobilenetv3'
        self.ensemble_weight = 0.10

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def freeze_backbone(self):
        """Freeze all backbone layers. Only classifier trains."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all layers for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_parameter_groups(self, base_lr: float = 0.001):
        """
        Returns parameter groups with differential learning rates.
        Backbone gets 10x lower LR than classifier during fine-tuning.
        """
        return [
            {'params': self.backbone.parameters(), 'lr': base_lr * 0.1},
            {'params': self.classifier.parameters(), 'lr': base_lr}
        ]


def build_mobilenetv3(num_classes: int = 2, pretrained: bool = True) -> MobileNetV3Lung:
    return MobileNetV3Lung(num_classes=num_classes, pretrained=pretrained)
