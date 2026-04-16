"""
NEMO Scan — Model 5: Vision Transformer ViT-B/16
Location: core/models/lung/vit.py

Role: Global pattern detection. Architecturally different from all
other models in the ensemble — uses self-attention instead of
convolutions. This diversity is valuable: ViT catches relationships
between distant regions of the X-ray that CNNs tend to miss.

Architecture: Image is divided into 16x16 pixel patches. These patches
are treated like words in a sentence. Self-attention finds relationships
between all patch pairs simultaneously — e.g. bilateral lung involvement
where both lungs show opacity at the same time.

Expected accuracy: 91-93%
Ensemble weight: 0.17
"""

import torch
import torch.nn as nn
import timm


class ViTB16Lung(nn.Module):
    """
    Vision Transformer ViT-B/16 fine-tuned for pneumonia detection.
    The only non-CNN model in the ensemble. Provides architectural
    diversity which improves overall ensemble accuracy.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()

        self.backbone = timm.create_model(
            'vit_base_patch16_224',
            pretrained=pretrained,
            num_classes=0,
            global_pool='token'     # use CLS token for classification
        )

        feature_dim = self.backbone.num_features  # 768 for ViT-B

        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),  # LayerNorm matches ViT's internal norm style
            nn.Dropout(p=0.3),
            nn.Linear(feature_dim, 256),
            nn.GELU(),              # GELU matches ViT's internal activations
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

        self.model_name = 'vit_b16'
        self.ensemble_weight = 0.17

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def freeze_backbone(self):
        """
        For ViT, freeze all but the last 2 transformer blocks.
        Freezing all layers makes ViT underperform on medical data.
        """
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False

        # Unfreeze last 2 blocks
        for name, param in self.backbone.named_parameters():
            if 'blocks.10' in name or 'blocks.11' in name or 'norm' in name:
                param.requires_grad = True

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_parameter_groups(self, base_lr: float = 0.001):
        """
        ViT requires very conservative fine-tuning LR.
        Transformers are sensitive to large learning rates.
        """
        return [
            {'params': self.backbone.parameters(), 'lr': base_lr * 0.01},
            {'params': self.classifier.parameters(), 'lr': base_lr}
        ]


def build_vit_b16(num_classes: int = 2, pretrained: bool = True) -> ViTB16Lung:
    return ViTB16Lung(num_classes=num_classes, pretrained=pretrained)
