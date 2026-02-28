import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
import numpy as np
from typing import Tuple

# 论文超参数
WS = 0.06                # CSSIM权重
WA = 0.002               # 各向异性损失权重
HVS_KERNEL_SIZE = 11     # HVS滤波器尺寸
HVS_SIGMA = 2.0          # HVS高斯核标准差
EPS = 1e-12

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    """生成二维高斯核，归一化后返回 (1,1,size,size)"""
    center = size // 2
    y = torch.arange(size, dtype=torch.float32) - center
    x = torch.arange(size, dtype=torch.float32) - center
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    kernel = torch.exp(-0.5 * (xx**2 + yy**2) / sigma**2)
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, size, size)


def _radial_masks(h: int, w: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """生成频域径向距离图和圆环索引图（用于RAPSD）"""
    y = torch.linspace(-(h//2), h//2 - (1 if h%2==0 else 0), h)
    x = torch.linspace(-(w//2), w//2 - (1 if w%2==0 else 0), w)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    dist = torch.sqrt(xx**2 + yy**2)
    max_dist = int(torch.max(dist).item())
    bin_mask = torch.clamp(dist.int(), 0, max_dist)
    # 中心化后返回（距离图和掩码都需要shift以匹配FFTshift后的频谱）
    dist = fft.fftshift(dist)
    bin_mask = fft.fftshift(bin_mask)
    return dist, bin_mask


class HalftoneMARLLoss(nn.Module):
    """
    MARL损失 L_MARL = - Σ_a Σ_{h'_a} R({h'_a, h_{-a}}, c) π_a(h'_a | c, z; θ)
    其中奖励 R = -MSE(HVS(h), HVS(c)) + ws * CSSIM(h, c)
    """
    def __init__(self, ws: float = WS):
        super().__init__()
        self.ws = ws
        self.register_buffer('hvs_kernel', _gaussian_kernel(HVS_KERNEL_SIZE, HVS_SIGMA))

    def _hvs_filter(self, x: torch.Tensor) -> torch.Tensor:
        """应用HVS低通滤波，保持尺寸不变"""
        padding = HVS_KERNEL_SIZE // 2
        kernel = self.hvs_kernel.to(device=DEVICE, dtype=x.dtype)
        return F.conv2d(x, kernel, padding=padding)

    def _cssim(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """计算对比度加权SSIM (CSSIM)，式(3-23)"""
        # 原始SSIM分量
        mu_h = self._hvs_filter(h)
        mu_c = self._hvs_filter(c)
        sigma_h_sq = self._hvs_filter(h ** 2) - mu_h ** 2
        sigma_c_sq = self._hvs_filter(c ** 2) - mu_c ** 2
        sigma_hc = self._hvs_filter(h * c) - mu_h * mu_c

        C1 = (0.01 * 1) ** 2
        C2 = (0.03 * 1) ** 2

        l = (2 * mu_h * mu_c + C1) / (mu_h ** 2 + mu_c ** 2 + C1)
        c_map = (2 * torch.sqrt(sigma_h_sq * sigma_c_sq + EPS) + C2) / (sigma_h_sq + sigma_c_sq + C2)
        s_map = (2 * sigma_hc + C2) / (torch.sqrt(sigma_h_sq * sigma_c_sq) + C2 + EPS)
        ssim = l * c_map * s_map

        # 局部对比度图 σ_c (式3-22)
        k = 2
        mu_c_local = self._hvs_filter(c)
        c_var = self._hvs_filter((c - mu_c_local) ** 2)
        c_contrast = k * torch.sqrt(c_var + EPS)
        c_contrast = torch.clamp(c_contrast, 0, 1)

        # 加权得到CSSIM
        cssim = c_contrast * ssim + (1 - c_contrast) * 1.0
        return cssim.mean()

    def _reward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """计算全局奖励 R = -MSE(HVS(h), HVS(c)) + ws * CSSIM"""
        h_hvs = self._hvs_filter(h)
        c_hvs = self._hvs_filter(c)
        mse = F.mse_loss(h_hvs, c_hvs)
        cssim = self._cssim(h, c)
        return -mse + self.ws * cssim

    def forward(self, prob: torch.Tensor, c: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        prob: 策略网络输出的白色概率图 [B,1,H,W]
        c:    连续调图像 [B,1,H,W]
        z:    噪声图（未直接使用，但保留接口）
        return: MARL损失标量
        """
        B, _, H, W = prob.shape
        # 从概率图采样一个二值图像h（作为h_{-a}的基础）
        h_sampled = torch.bernoulli(prob).detach()  # [B,1,H,W]

        # 预滤波c，后续奖励计算可以复用（但简单循环中仍会重复计算，此处保留可优化空间）
        c_hvs = self._hvs_filter(c)  # 未使用，但可留作将来优化

        total_loss = 0.0
        # 遍历每个样本、每个像素
        for b in range(B):
            for i in range(H):
                for j in range(W):
                    # 对当前像素的两种动作
                    for action in (0.0, 1.0):
                        # 构建h'：将h_sampled[b,0,i,j]替换为action
                        h_prime = h_sampled[b:b+1, :, :, :].clone()  # [1,1,H,W]
                        h_prime[0, 0, i, j] = action
                        # 计算奖励
                        reward = self._reward(h_prime, c[b:b+1])
                        # 当前动作的概率
                        if action == 1.0:
                            pi_a = prob[b, 0, i, j]
                        else:
                            pi_a = 1.0 - prob[b, 0, i, j]
                        total_loss -= pi_a * reward

        # 平均损失（除以总像素数）
        return total_loss / (B * H * W)


class AnisotropySuppressionLoss(nn.Module):
    """
    各向异性抑制损失 L_AS = Σ_{f_ρ} Σ_{f∈r(f_ρ)} (P̂_θ(f) - P_θ(f_ρ))^2
    其中 P̂_θ(f) = 1/N |DFT(π(h=1|c_g, z_g; θ))|^2，P_θ(f_ρ)是径向平均
    """
    def __init__(self, wa: float = WA):
        super().__init__()
        self.wa = wa

    def _power_spectrum(self, prob: torch.Tensor) -> torch.Tensor:
        """计算功率谱 P̂(f) = |DFT(prob)|² / N, prob为概率图 [B,1,H,W]"""
        B, C, H, W = prob.shape
        assert C == 1, "概率图必须为单通道"
        prob32 = prob.squeeze(1).to(dtype=torch.float32)  # [B,H,W]
        fft_vals = fft.fft2(prob32, dim=(-2, -1))
        fft_vals = fft.fftshift(fft_vals)

        N = H * W
        power = torch.abs(fft_vals) ** 2 / N
        return power  # [B,H,W]

    def _radial_average(self, power: torch.Tensor, bin_mask: torch.Tensor) -> torch.Tensor:
        """计算径向平均功率谱 P(f_ρ)"""
        B, H, W = power.shape
        max_bin = bin_mask.max().item()
        radial = torch.zeros(B, max_bin + 1, device=power.device)
        for b in range(B):
            for r in range(max_bin + 1):
                mask = (bin_mask == r)
                if mask.sum() == 0:
                    continue
                radial[b, r] = power[b, mask].mean()
        return radial

    def forward(self, prob_cg: torch.Tensor) -> torch.Tensor:
        """
        prob_cg: 恒定灰度图的输出概率图 [B,1,H,W]
        return: 各向异性损失（乘以wa后）
        """
        B, C, H, W = prob_cg.shape
        # 生成径向掩码（与当前图像尺寸匹配）
        _, bin_mask = _radial_masks(H, W)
        bin_mask = bin_mask.to(prob_cg.device)

        # 功率谱
        power = self._power_spectrum(prob_cg)  # [B,H,W]

        # 径向平均
        radial = self._radial_average(power, bin_mask)  # [B, max_bin+1]

        # 将径向平均扩展到原图尺寸
        radial_expanded = radial[:, bin_mask.flatten()].reshape(B, H, W)

        # 计算损失：各点与径向平均的平方差之和
        loss = torch.pow(power - radial_expanded + EPS, 2).sum(dim=(-2, -1)).mean()
        return self.wa * loss


class TotalHalftoneLoss(nn.Module):
    """
    总损失 L_total = L_MARL + w_a * L_AS
    """
    def __init__(self, ws: float = WS, wa: float = WA):
        super().__init__()
        self.marl_loss = HalftoneMARLLoss(ws)
        self.as_loss = AnisotropySuppressionLoss(wa)

    def forward(self, prob: torch.Tensor, c: torch.Tensor, z: torch.Tensor, prob_cg: torch.Tensor):
        """
        prob:    普通图的概率图 [B,1,H,W]
        c:       普通连续调图 [B,1,H,W]
        z:       高斯噪声 [B,1,H,W] (传递给MARL损失，但未使用)
        prob_cg: 恒定灰度图的概率图 [B,1,H,W]
        return: (total_loss, marl_loss, as_loss)
        """
        marl = self.marl_loss(prob, c, z)
        as_ = self.as_loss(prob_cg)
        total = marl + as_
        return total, marl, as_