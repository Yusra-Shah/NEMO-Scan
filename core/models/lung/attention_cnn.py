"""
NEMO Scan — Model 7: Attention CNN (Custom Architecture)
Location: core/models/lung/attention_cnn.py

Role: Explainability model. The ONLY model with a Grad-CAM head.
Produces the heatmap overlay shown in the GUI scan panel.
Does NOT participate in ensemble voting (weight = 0.00).
Runs in parallel with the ensemble and outputs a visualization.

Architecture: Custom convolutional network with spatial attention gates.
Attention gates learn to suppress irrelevant background regions
(diaphragm, bones, soft tissue) and focus on lung parenchyma.
Grad-CAM hooks are attached to the final conv layer. When the model
makes a prediction, gradients flow back to the feature maps —
regions with highest activation are highlighted red in the heatmap.

Expected accuracy: 90-92%
Ensemble weight: 0.00 (explainability only)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """
    Spatial attention gate.
    Learns to assign higher weight to diagnostically relevant regions.
    Applied after each convolutional block.
    """

    def __init__(self, in_channels: int, gate_channels: int):
        super().__init__()

        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels + gate_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        # Upsample gate to match x spatial dimensions
        gate_up = F.interpolate(gate, size=x.shape[2:], mode='bilinear', align_corners=False)
        combined = torch.cat([x, gate_up], dim=1)
        attention_map = self.gate_conv(combined)
        return x * attention_map


class ConvBlock(nn.Module):
    """Standard Conv-BN-ReLU block."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionCNN(nn.Module):
    """
    Custom Attention CNN for X-ray explainability.

    The final_conv layer is where Grad-CAM hooks are attached.
    Do not rename or remove this layer — gradcam.py depends on it.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()

        # Encoder — progressive feature extraction
        self.enc1 = nn.Sequential(
            ConvBlock(3, 32, 3),
            ConvBlock(32, 32, 3)
        )                               # 224 -> 224

        self.pool1 = nn.MaxPool2d(2)    # 224 -> 112

        self.enc2 = nn.Sequential(
            ConvBlock(32, 64, 3),
            ConvBlock(64, 64, 3)
        )                               # 112 -> 112

        self.pool2 = nn.MaxPool2d(2)    # 112 -> 56

        self.enc3 = nn.Sequential(
            ConvBlock(64, 128, 3),
            ConvBlock(128, 128, 3)
        )                               # 56 -> 56

        self.pool3 = nn.MaxPool2d(2)    # 56 -> 28

        self.enc4 = nn.Sequential(
            ConvBlock(128, 256, 3),
            ConvBlock(256, 256, 3)
        )                               # 28 -> 28

        self.pool4 = nn.MaxPool2d(2)    # 28 -> 14

        # Bottleneck — deepest feature representation
        # THIS IS THE GRAD-CAM TARGET LAYER — do not rename
        self.final_conv = nn.Sequential(
            ConvBlock(256, 512, 3),
            ConvBlock(512, 512, 3),
            ConvBlock(512, 512, 3)
        )                               # 14 -> 14

        # Attention gates — suppress background, highlight lungs
        self.attn1 = AttentionGate(in_channels=256, gate_channels=512)
        self.attn2 = AttentionGate(in_channels=128, gate_channels=512)

        # Global average pooling + classification head
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

        self.model_name = 'attention_cnn'
        self.ensemble_weight = 0.00     # explainability only, not in voting

        # Initialize weights
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck (Grad-CAM target)
        bottleneck = self.final_conv(self.pool4(e4))

        # Apply attention gates using bottleneck as gate signal
        a4 = self.attn1(e4, bottleneck)
        a3 = self.attn2(e3, bottleneck)

        # Pool bottleneck for classification
        pooled = self.global_pool(bottleneck)
        return self.classifier(pooled)

    def get_gradcam_target_layer(self) -> nn.Module:
        """
        Returns the layer used as Grad-CAM target.
        Called by gradcam.py to attach gradient hooks.
        Always returns the last conv in final_conv block.
        """
        return self.final_conv[-1].block[0]  # last Conv2d in bottleneck

    def freeze_backbone(self):
        """Freeze encoder layers, train only bottleneck and classifier."""
        for layer in [self.enc1, self.enc2, self.enc3, self.enc4]:
            for param in layer.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.parameters():
            param.requires_grad = True

    def get_parameter_groups(self, base_lr: float = 0.001):
        encoder_params = (
            list(self.enc1.parameters()) +
            list(self.enc2.parameters()) +
            list(self.enc3.parameters()) +
            list(self.enc4.parameters())
        )
        head_params = (
            list(self.final_conv.parameters()) +
            list(self.attn1.parameters()) +
            list(self.attn2.parameters()) +
            list(self.classifier.parameters())
        )
        return [
            {'params': encoder_params, 'lr': base_lr * 0.1},
            {'params': head_params, 'lr': base_lr}
        ]

    def _initialize_weights(self):
        """Kaiming initialization for conv layers, Xavier for linear."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)


def build_attention_cnn(num_classes: int = 2) -> AttentionCNN:
    """
    Note: AttentionCNN has no pretrained weights — it is trained from scratch.
    This is intentional. It is a custom architecture and no ImageNet
    pretrained version exists. It will converge well with our 15,000 images.
    """
    return AttentionCNN(num_classes=num_classes)
