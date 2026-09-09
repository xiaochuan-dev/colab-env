# model/backbone.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cfg


class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth)
        self.conv2 = nn.Conv2d(4 * growth, growth, 3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        return torch.cat([x, out], dim=1)


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(_DenseLayer(in_channels + i * growth, growth))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class _Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pool = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        x = self.conv(F.relu(self.bn(x), inplace=True))
        return self.pool(x)


class DenseNetFeature(nn.Module):
    """简化版 DenseNet，输出 2D feature map (B, C, H', W')"""
    def __init__(self):
        super().__init__()
        growth = cfg.DENSENET_GROWTH
        # stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        channels = 64
        blocks = []
        for i, n_layers in enumerate(cfg.DENSENET_BLOCKS):
            blocks.append(_DenseBlock(n_layers, channels, growth))
            channels += n_layers * growth
            if i != len(cfg.DENSENET_BLOCKS) - 1:
                out_c = int(channels * cfg.DENSENET_THETA)
                blocks.append(_Transition(channels, out_c))
                channels = out_c
        self.blocks = nn.Sequential(*blocks)
        self.out_channels = channels
        self.final_bn = nn.BatchNorm2d(channels)

        # 投影到 d_model
        self.proj = nn.Conv2d(channels, cfg.D_MODEL, 1)

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.stem(x)
        x = self.blocks(x)
        x = F.relu(self.final_bn(x), inplace=True)
        x = self.proj(x)          # (B, d_model, h, w)
        return x
