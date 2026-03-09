import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self._init_weights()

    def _init_weights(self):
        """残差块初始化：严格对齐论文N(0,0.01²) + 偏置0"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        residual = x
        # 预激活结构，保证梯度畅通
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = out + residual
        return out

class HalftoningPolicyNet(nn.Module):
    """
    半色调MARL策略网络（全卷积）100%对齐论文
    输入：连续调图像c + 高斯噪声z（拼接为2通道）
    输出：每个像素选择1的概率π(h=1|c,z;θ)
    """
    def __init__(self, in_channels=2, out_channels=1, base_channels=32, num_blocks=16):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.blocks = nn.Sequential(*[ResidualBlock(base_channels) for _ in range(num_blocks)])
        self.final = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self._init_network_weights()

    def _init_network_weights(self):
        """全局初始化：严格对齐论文N(0,0.01²)，所有偏置为0"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, cont_img, noise_img=None, noise_std=1.0):
        """
        严格对齐论文：输入为连续调图像+高斯噪声拼接
        :param cont_img: 连续调图像 [B,1,H,W]
        :param noise_img: 高斯噪声 [B,1,H,W]，为None时自动生成
        :param noise_std: 噪声标准差，论文默认1.0
        :return: 每个像素为1的概率 [B,1,H,W]
        """
        if cont_img.dim() == 3:
            cont_img = cont_img.unsqueeze(0)

        # 噪声处理：固定输入噪声，修复LAS损失不收敛问题
        if noise_img is None:
            noise_img = torch.randn_like(cont_img, device=cont_img.device, dtype=cont_img.dtype) * noise_std
        else:
            if noise_img.dim() == 3:
                noise_img = noise_img.unsqueeze(0)
            noise_img = noise_img.to(cont_img.device, dtype=cont_img.dtype, non_blocking=True)

        # 拼接输入，保证内存连续
        x = torch.cat([cont_img, noise_img], dim=1).contiguous()

        # 前向传播
        x = self.initial(x)
        x = self.blocks(x)
        x = self.final(x)
        prob = self.sigmoid(x)

        return prob