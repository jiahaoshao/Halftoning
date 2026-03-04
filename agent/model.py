import torch
import torch.nn as nn
import torch.nn.functional as F
from agent.loss import EPS, PROB_CLAMP_MIN, PROB_CLAMP_MAX

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self._init_weights()

    def _init_weights(self):
        """残差块内卷积/BN层初始化：对齐N(0,0.01²) + 偏置0"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 卷积核：N(0, 0.01²) 初始化
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                # 偏置（若有）强制设为0
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                # BN层：weight=1，bias=0（标准初始化）
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        residual = x
        # 预激活结构：BN→ReLU→Conv，更适配深层残差网络
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = out + residual  # 残差路径无激活，保证梯度畅通
        return out

class HalftoningPolicyNet(nn.Module):
    """
    半色调MARL策略网络（全卷积）
    核心输入：连续调图像 + 随机高斯噪声图像（拼接为2通道）
    核心输出：每个像素（智能体）选择为白色（1）的概率，黑色为0
    初始化规则：所有卷积核N(0,0.01²)，所有偏置0
    """
    def __init__(self, in_channels=2, out_channels=1, base_channels=32, num_blocks=16):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.blocks = nn.Sequential(*[ResidualBlock(base_channels) for _ in range(num_blocks)])
        self.final = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self._init_network_weights()

    def _init_network_weights(self):
        """
        全局初始化：所有卷积核N(0,0.01²)，所有偏置（含BN）=0
        覆盖初始卷积/最终卷积/残差块外的所有层
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 核心规则：卷积权重N(0, 0.01²)
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                # 偏置（若有）强制为0
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                # BN层：weight=1，bias=0（保证分布稳定）
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, cont_img, noise_img=None, noise_std=1.0):
        # 1. 维度校验与补全（适配单样本输入）
        if cont_img.dim() == 3:  # (1,H,W) → (1,1,H,W)
            cont_img = cont_img.unsqueeze(0)
        # 2. 生成/校验噪声图像
        if noise_img is None:
            noise_img = torch.randn_like(cont_img) * noise_std  # 与连续调图同尺寸的高斯噪声
        else:
            if noise_img.dim() == 3:
                noise_img = noise_img.unsqueeze(0)
        # 【优化】保证tensor内存连续，加速后续cat和卷积
        noise_img = (noise_img - noise_img.min()) / (noise_img.max() - noise_img.min() + EPS)
        x = torch.cat([cont_img, noise_img], dim=1).contiguous()
        x = self.initial(x)
        x = self.blocks(x)
        x = self.final(x)
        prob = self.sigmoid(x)  # 白色概率
        prob = torch.clamp(prob, PROB_CLAMP_MIN, PROB_CLAMP_MAX) # 防止概率极端值（梯度消失）
        return prob