import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out

class PolicyNetwork(nn.Module):
    """
    全卷积策略网络，输入2通道（连续调图+噪声），输出每个像素为白色的概率
    """
    def __init__(self, in_channels=2, out_channels=1, base_channels=32, num_blocks=8):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU()
        )
        self.blocks = nn.Sequential(*[ResidualBlock(base_channels) for _ in range(num_blocks)])
        self.final = nn.Conv2d(base_channels, out_channels, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, c, z):
        # c: (B,1,H,W), z: (B,1,H,W)
        x = torch.cat([c, z], dim=1)
        x = self.initial(x)
        x = self.blocks(x)
        x = self.final(x)
        prob = self.sigmoid(x)  # 白色概率
        return prob