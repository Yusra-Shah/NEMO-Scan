"""
PneumoScan - AttentionCNN Architecture (Inference Copy)
Location: core/inference/attention_arch.py

This is the architecture definition used by the inference engine
to reconstruct the AttentionCNN model before loading trained weights.
Must match the architecture used during training exactly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention gate."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.fc  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg   = self.fc(self.gap(x))
        mx    = self.fc(self.gmp(x))
        scale = self.sigmoid(avg + mx).unsqueeze(-1).unsqueeze(-1)
        return x * scale


class ConvBnRelu(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionCNN(nn.Module):
    """
    Custom CNN with channel attention gates.
    Trained from scratch on chest X-ray data.
    Used for Grad-CAM heatmap generation only.
    The final_conv layer is the Grad-CAM target.

    Architecture must exactly match what was used during Kaggle training.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            ConvBnRelu(3, 32, 3, 2, 1),    # 224 -> 112
            ConvBnRelu(32, 64, 3, 1, 1),   # 112 -> 112
        )

        # Block 1
        self.block1 = nn.Sequential(
            ConvBnRelu(64, 128, 3, 2, 1),  # 112 -> 56
            ConvBnRelu(128, 128, 3, 1, 1),
        )
        self.att1 = ChannelAttention(128)

        # Block 2
        self.block2 = nn.Sequential(
            ConvBnRelu(128, 256, 3, 2, 1), # 56 -> 28
            ConvBnRelu(256, 256, 3, 1, 1),
            ConvBnRelu(256, 256, 3, 1, 1),
        )
        self.att2 = ChannelAttention(256)

        # Block 3
        self.block3 = nn.Sequential(
            ConvBnRelu(256, 512, 3, 2, 1), # 28 -> 14
            ConvBnRelu(512, 512, 3, 1, 1),
            ConvBnRelu(512, 512, 3, 1, 1),
        )
        self.att3 = ChannelAttention(512)

        # Final conv - Grad-CAM target layer
        self.final_conv = nn.Sequential(
            ConvBnRelu(512, 512, 3, 1, 1), # 14 -> 14
            ConvBnRelu(512, 512, 3, 1, 1),
        )

        # Classifier head
        self.gap        = nn.AdaptiveAvgPool2d(1)
        self.dropout    = nn.Dropout(0.4)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.att1(self.block1(x))
        x = self.att2(self.block2(x))
        x = self.att3(self.block3(x))
        x = self.final_conv(x)
        x = self.gap(x)
        x = self.dropout(x.flatten(1))
        return self.classifier(x)

    def get_gradcam_target_layer(self) -> nn.Module:
        """
        Returns the Conv2d layer used as Grad-CAM target.
        This is the last Conv2d inside the last ConvBnRelu of final_conv.
        """
        return self.final_conv[-1].block[0]
