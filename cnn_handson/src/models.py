"""
Model definitions for the brain-MRI CNN hands-on.

Covers three task types referenced in the course (Training_Course.pdf p.92,
"Types of models"):

  1. Image classification   -> SimpleMLP, SimpleCNN, DeeperCNN
  2. Semantic segmentation  -> UNet
  3. Object detection       -> SimpleDetector (single-object: presence + bbox)

All models target single-channel (grayscale) 128 x 128 brain MRI slices.

Output-size formula from lecture (p.87):
    O = floor((I + 2P - F) / S) + 1
"""

from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_IMG_SIZE = 128


def count_parameters(model: nn.Module) -> int:
    """Total trainable parameters (matches the 'model complexity' note on p.91)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =====================================================================
# 1. Classification models
# =====================================================================

class SimpleMLP(nn.Module):
    """Flat MLP baseline that destroys spatial structure (lecture p.82)."""

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (1, DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE),
        num_classes: int = 4,
        hidden: int = 256,
    ):
        super().__init__()
        in_features = input_shape[0] * input_shape[1] * input_shape[2]
        self.net = nn.Sequential(OrderedDict([
            ("flatten", nn.Flatten()),
            ("fc1", nn.Linear(in_features, hidden)),
            ("relu1", nn.ReLU(inplace=True)),
            ("dropout", nn.Dropout(0.3)),
            ("fc2", nn.Linear(hidden, num_classes)),
        ]))

    def forward(self, x):
        return self.net(x)


class SimpleCNN(nn.Module):
    """
    The 'textbook' small CNN: 3 Conv-ReLU-Pool blocks followed by an MLP head.

    Input  : 1 x 128 x 128
    conv1  : Conv2d(1, 16, k=3, p=1)   -> 16 x 128 x 128
    pool1  : MaxPool2d(2)              -> 16 x  64 x  64
    conv2  : Conv2d(16, 32, k=3, p=1)  -> 32 x  64 x  64
    pool2  : MaxPool2d(2)              -> 32 x  32 x  32
    conv3  : Conv2d(32, 64, k=3, p=1)  -> 64 x  32 x  32
    pool3  : MaxPool2d(2)              -> 64 x  16 x  16
    flatten                            -> 16,384
    fc1    : Linear(16384, 128) + ReLU + Dropout
    fc2    : Linear(128, num_classes)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        img_size: int = DEFAULT_IMG_SIZE,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.img_size = img_size

        self.features = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)),
            ("relu1", nn.ReLU(inplace=True)),
            ("pool1", nn.MaxPool2d(2)),
            ("conv2", nn.Conv2d(16, 32, kernel_size=3, padding=1)),
            ("relu2", nn.ReLU(inplace=True)),
            ("pool2", nn.MaxPool2d(2)),
            ("conv3", nn.Conv2d(32, 64, kernel_size=3, padding=1)),
            ("relu3", nn.ReLU(inplace=True)),
            ("pool3", nn.MaxPool2d(2)),
        ]))

        feat_side = img_size // 8
        feat_dim = 64 * feat_side * feat_side

        self.classifier = nn.Sequential(OrderedDict([
            ("flatten", nn.Flatten()),
            ("dropout", nn.Dropout(dropout)),
            ("fc1", nn.Linear(feat_dim, 128)),
            ("relu", nn.ReLU(inplace=True)),
            ("fc2", nn.Linear(128, num_classes)),
        ]))

    def forward(self, x):
        return self.classifier(self.features(x))

    @torch.no_grad()
    def feature_maps(self, x):
        """Return intermediate activations for visualization."""
        acts = {}
        out = self.features.conv1(x);  acts["conv1"] = out.clone()
        out = self.features.relu1(out)
        out = self.features.pool1(out); acts["pool1"] = out.clone()
        out = self.features.conv2(out); acts["conv2"] = out.clone()
        out = self.features.relu2(out)
        out = self.features.pool2(out); acts["pool2"] = out.clone()
        out = self.features.conv3(out); acts["conv3"] = out.clone()
        out = self.features.relu3(out)
        out = self.features.pool3(out); acts["pool3"] = out.clone()
        return acts


class _ResidualBlock(nn.Module):
    """Residual block with BatchNorm. Pre-activation style for stability."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class DeeperCNN(nn.Module):
    """
    A more sophisticated CNN: BatchNorm + residual blocks + global average pooling.

    Same input/output shapes as SimpleCNN but ~5-10x deeper.
    Demonstrates the standard upgrades that take a 'textbook CNN' closer to a
    modern architecture (mini-ResNet style).

    Stages (input 1 x 128 x 128):
      stem    : Conv 1->32, BN, ReLU       ->  32 x 128 x 128
      stage1  : 2x ResBlock(32, 32)        ->  32 x 128 x 128
      stage2  : ResBlock(32, 64,  s=2) + ResBlock(64, 64)   ->  64 x 64 x 64
      stage3  : ResBlock(64, 128, s=2) + ResBlock(128, 128) -> 128 x 32 x 32
      stage4  : ResBlock(128,256, s=2) + ResBlock(256, 256) -> 256 x 16 x 16
      gap     : AdaptiveAvgPool2d(1)       -> 256
      fc      : Linear(256, num_classes)
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(_ResidualBlock(32, 32),  _ResidualBlock(32, 32))
        self.stage2 = nn.Sequential(_ResidualBlock(32, 64,  stride=2), _ResidualBlock(64, 64))
        self.stage3 = nn.Sequential(_ResidualBlock(64, 128, stride=2), _ResidualBlock(128, 128))
        self.stage4 = nn.Sequential(_ResidualBlock(128, 256, stride=2), _ResidualBlock(256, 256))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


# =====================================================================
# 2. Segmentation: U-Net
# =====================================================================

class _DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) x 2 -- the basic unit of U-Net."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    A small U-Net for binary tumor segmentation.

    Encoder path (4 downsampling steps):
        128 -> 64 -> 32 -> 16 -> 8   (channels: 32 -> 64 -> 128 -> 256)
    Bottleneck:
        8x8 with 512 channels
    Decoder path (4 upsampling steps, with skip connections):
        8 -> 16 -> 32 -> 64 -> 128
    Output: 1-channel logits (apply sigmoid -> binary mask).

    Reference: Ronneberger et al., "U-Net: Convolutional Networks for
    Biomedical Image Segmentation", MICCAI 2015.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base: int = 32):
        super().__init__()
        c1, c2, c3, c4, c5 = base, base * 2, base * 4, base * 8, base * 16

        # Encoder
        self.enc1 = _DoubleConv(in_channels, c1)
        self.enc2 = _DoubleConv(c1, c2)
        self.enc3 = _DoubleConv(c2, c3)
        self.enc4 = _DoubleConv(c3, c4)

        # Bottleneck
        self.bottleneck = _DoubleConv(c4, c5)

        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.up4 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.dec4 = _DoubleConv(c5, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = _DoubleConv(c4, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = _DoubleConv(c3, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = _DoubleConv(c2, c1)

        self.out_conv = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)         # logits, shape [B, out_channels, H, W]


# =====================================================================
# 3. Detection: single-object bbox + presence
# =====================================================================

class SimpleDetector(nn.Module):
    """
    A minimal single-object detector.

    Backbone: a SimpleCNN-style feature extractor with global average pooling.
    Two heads:
        - presence : 1 logit  (BCEWithLogits -> sigmoid -> probability of tumor)
        - bbox     : 4 values (normalized cx, cy, w, h in [0, 1])

    For training, the bbox loss is only applied to images that actually
    contain a tumor (presence = 1), so the model isn't forced to regress
    coordinates from empty slices.
    """

    def __init__(self, in_channels: int = 1, dropout: float = 0.3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),                # -> [B, 256, 1, 1]
        )

        self.dropout = nn.Dropout(dropout)

        # Presence head: is there a tumor in this slice?
        self.presence_head = nn.Linear(256, 1)

        # Bbox head: (cx, cy, w, h) in [0, 1]. We apply sigmoid in forward.
        self.bbox_head = nn.Linear(256, 4)

    def forward(self, x):
        feat = self.backbone(x).flatten(1)         # [B, 256]
        feat = self.dropout(feat)
        presence_logit = self.presence_head(feat).squeeze(-1)   # [B]
        bbox = torch.sigmoid(self.bbox_head(feat))              # [B, 4] in [0,1]
        return presence_logit, bbox
