import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return self.relu(out)


class DRLHalftoneNet(nn.Module):
    def __init__(self, num_res_blocks=16, channels=32):
        super().__init__()
        # 输入层：2通道（连续调+噪声）→ channels通道
        self.input_conv = nn.Conv2d(2, channels, 3, padding=1)
        self.input_bn = nn.BatchNorm2d(channels)
        self.input_relu = nn.ReLU()
        # 残差块序列
        self.res_blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_res_blocks)])
        # 输出层：channels → 1通道（概率）
        self.output_conv = nn.Conv2d(channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, c, z):
        # c: (B,1,H,W), z: (B,1,H,W) → 拼接为(B,2,H,W)
        x = torch.cat([c, z], dim=1)
        x = self.input_conv(x)
        x = self.input_bn(x)
        x = self.input_relu(x)
        x = self.res_blocks(x)
        prob = self.sigmoid(self.output_conv(x))  # (B,1,H,W)
        return prob