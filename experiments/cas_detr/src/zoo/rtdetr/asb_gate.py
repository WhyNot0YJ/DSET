"""ASB-Gate module for channel compression before decoder."""

import torch.nn as nn


class ASBGate(nn.Module):
    """Asymmetric Semantic Bottleneck Gate (ASB-Gate)."""

    def __init__(self, in_channels: int = 256, out_channels: int = 128, mid_channels: int = 64):
        super().__init__()
        self.semantic_path = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(in_channels, mid_channels),
            nn.GELU(),
            nn.Linear(mid_channels, in_channels),
            nn.Sigmoid()
        )
        self.spatial_path = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels
        )
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        weights = self.semantic_path(x).unsqueeze(-1).unsqueeze(-1)
        x = x * weights
        x = self.spatial_path(x)
        return self.proj(x)

