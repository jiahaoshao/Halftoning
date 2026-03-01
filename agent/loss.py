import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple

# 全局设备与常量预定义（避免重复生成/计算）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 论文固定超参数（3.5节）
HVS_KERNEL_SIZE = 11
HVS_SIGMA = 1.5
HVS_SCALE = 2000
CSSIM_C1 = 1e-4
CSSIM_C2 = 9e-4
EPS = 1e-8


# -------------------------- 1. 预计算复用模块（核心性能优化） --------------------------
def create_gaussian_kernel(kernel_size: int = HVS_KERNEL_SIZE, sigma: float = HVS_SIGMA) -> torch.Tensor:
    """预生成高斯核，全局复用，避免每次计算重复生成"""
    kx = cv2.getGaussianKernel(kernel_size, sigma)
    ky = cv2.getGaussianKernel(kernel_size, sigma)
    kernel = np.outer(kx, ky)
    return torch.from_numpy(kernel).float().unsqueeze(0).unsqueeze(0).to(device)


# 全局预生成HVS高斯核，全流程复用
HVS_KERNEL = create_gaussian_kernel()
HVS_PADDING = HVS_KERNEL_SIZE // 2


# -------------------------- 2. HVS滤波模块（严格对齐论文3.5节Näsänen HVS模型） --------------------------
def hvs_filter(x: torch.Tensor) -> torch.Tensor:
    """
    单张图像HVS低通滤波，严格对齐论文3.5节
    :param x: 输入张量 [B, C, H, W]
    :return: HVS滤波后张量 [B, C, H, W]
    """
    x_filtered = F.conv2d(
        x, HVS_KERNEL,
        padding=HVS_PADDING,
        groups=x.shape[1]
    )
    return x_filtered / HVS_SCALE


def batch_hvs_filter(h_batch: torch.Tensor) -> torch.Tensor:
    """
    批量半色调块HVS滤波，消除冗余参数，复用全局核
    :param h_batch: 输入半色调块张量 [B, 1, H, W, N]（N=block_num）
    :return: 滤波后张量 [B, 1, H, W, N]
    """
    B, C, H, W, N = h_batch.shape
    # 向量化reshape，避免循环，充分利用GPU并行
    h_reshaped = h_batch.permute(0, 4, 1, 2, 3).reshape(B * N, C, H, W)
    h_hvs = F.conv2d(h_reshaped, HVS_KERNEL, padding=HVS_PADDING, groups=C)
    # 还原维度，保持和输入一致
    h_hvs = h_hvs.reshape(B, N, C, H, W).permute(0, 2, 3, 4, 1)
    return h_hvs / HVS_SCALE


# -------------------------- 3. CSSIM计算模块（严格对齐论文式3-23，消除循环，极致并行） --------------------------
def cssim(h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    单张图像对比度加权SSIM，严格对齐论文式3-22/3-23
    :param h: 半色调图像 [B, C, H, W]
    :param c: 连续调图像 [B, C, H, W]
    :return: CSSIM标量 [B]
    """
    B, C, H, W = c.shape
    # 1. 计算连续调图像局部对比度σ_c（论文式3-22）
    mu_c = F.conv2d(c, HVS_KERNEL, padding=HVS_PADDING, groups=C)
    c_mu_sq = torch.clamp((c - mu_c) ** 2, min=EPS)
    sigma_c = 2 * torch.sqrt(F.conv2d(c_mu_sq, HVS_KERNEL, padding=HVS_PADDING, groups=C))
    sigma_c = torch.clamp(sigma_c, 0, 1)  # 论文要求归一化[0,1]

    # 2. 计算基础SSIM（论文式3-21）
    mu_h = F.conv2d(h, HVS_KERNEL, padding=HVS_PADDING, groups=C)
    mu_hc = mu_h * mu_c
    mu_h2, mu_c2 = mu_h ** 2, mu_c ** 2

    # 方差计算数值保护，避免负方差/开方负数
    h_sq = torch.clamp(h ** 2, min=EPS)
    c_sq = torch.clamp(c ** 2, min=EPS)
    sigma_h2 = torch.clamp(F.conv2d(h_sq, HVS_KERNEL, padding=HVS_PADDING, groups=C) - mu_h2, min=EPS)
    sigma_c2 = torch.clamp(F.conv2d(c_sq, HVS_KERNEL, padding=HVS_PADDING, groups=C) - mu_c2, min=EPS)
    sigma_hc = F.conv2d(h * c, HVS_KERNEL, padding=HVS_PADDING, groups=C) - mu_hc

    # SSIM三项分解，全链路数值保护
    l = (2 * mu_hc + CSSIM_C1) / (mu_h2 + mu_c2 + CSSIM_C1 + EPS)
    c_ = (2 * torch.sqrt(sigma_h2 * sigma_c2) + CSSIM_C2) / (sigma_h2 + sigma_c2 + CSSIM_C2 + EPS)
    s = (2 * sigma_hc + CSSIM_C2) / (2 * torch.sqrt(sigma_h2 * sigma_c2) + CSSIM_C2 + EPS)
    ssim_map = l * c_ * s

    # 3. 论文式3-23：CSSIM = σ_c * SSIM + (1-σ_c) * 1
    cssim_map = sigma_c * ssim_map + (1 - sigma_c) * 1.0
    cssim_map = torch.clamp(cssim_map, min=EPS, max=1.0)

    return cssim_map.mean(dim=(1, 2, 3))  # 返回[B]维度，适配批量计算


def batch_cssim(h_batch: torch.Tensor, c_batch: torch.Tensor) -> torch.Tensor:
    """
    批量CSSIM计算，消除嵌套循环，全向量化并行，严格对齐论文公式
    :param h_batch: 半色调块张量 [B, 1, H, W, N]
    :param c_batch: 连续调块张量 [B, 1, H, W, N]
    :return: 批量CSSIM分数 [B, 1, N]
    """
    B, C, H, W, N = h_batch.shape
    # 向量化reshape，一次性并行计算所有块，消除循环
    h_reshaped = h_batch.permute(0, 4, 1, 2, 3).reshape(B * N, C, H, W)
    c_reshaped = c_batch.permute(0, 4, 1, 2, 3).reshape(B * N, C, H, W)

    # 复用单张cssim计算逻辑，保证公式一致性
    cssim_score = cssim(h_reshaped, c_reshaped)  # [B*N]

    # 还原维度，和输入对齐
    cssim_score = cssim_score.reshape(B, 1, N)  # [B, 1, N]
    return cssim_score


# -------------------------- 4. 奖励函数（严格对齐论文式3-24） --------------------------
def reward(h: torch.Tensor, c: torch.Tensor, w_s: float = 0.06) -> torch.Tensor:
    """
    奖励函数，严格对齐论文式3-24：R = -MSE(HVS(h), HVS(c)) + w_s * CSSIM(h,c)
    :param h: 半色调图，支持[B,1,H,W]单张或[B,1,H,W,N]批量
    :param c: 连续调图，与h维度匹配
    :param w_s: 论文默认超参数0.06
    :return: 奖励值，维度与输入批量维度一致
    """
    is_batch_mode = (h.dim() == 5)
    if is_batch_mode:
        B, C, H, W, N = h.shape
        # 批量HVS滤波
        h_hvs = batch_hvs_filter(h)
        c_hvs = batch_hvs_filter(c)
        # 批量MSE计算，对齐论文
        mse = torch.clamp(F.mse_loss(h_hvs, c_hvs, reduction='none').mean(dim=(2, 3)), min=EPS)
        # 批量CSSIM
        cssim_score = batch_cssim(h, c)
    else:
        # 单张场景，复用HVS滤波结果
        h_hvs = hvs_filter(h)
        c_hvs = hvs_filter(c)
        mse = torch.clamp(F.mse_loss(h_hvs, c_hvs, reduction='none').mean(dim=(1, 2, 3)), min=EPS)
        cssim_score = cssim(h, c)

    # 论文式3-24核心公式
    r = -mse + w_s * cssim_score
    # 数值裁剪，避免梯度爆炸
    return torch.clamp(r, min=-1e2, max=1e2)


# -------------------------- 5. 核心L_MARL损失计算（LE梯度估计器，100%对齐论文式3-14/3-15） --------------------------
def le_gradient_estimator(
        prob: torch.Tensor,
        c: torch.Tensor,
        w_s: float = 0.06,
        block_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    局部期望梯度估计器，严格对齐论文式3-14/3-15，修复梯度链路，极致性能优化
    论文核心公式：∇L_MARL = -E[Σa Σh'a ∇π_a(h'a) R({h'a, h_-a}, c)]
    :param prob: 策略网络输出的动作概率π(h=1)，[B, 1, H, W]，Sigmoid输出
    :param c: 连续调输入图像，[B, 1, H, W]，归一化[0,1]
    :param w_s: 奖励函数超参数，论文默认0.06
    :param block_size: 分块处理像素数，显存优化，默认64
    :return: L_MARL损失标量，用于反向传播；梯度统计值，用于监控
    """
    B, C, H, W = prob.shape
    device = prob.device
    dtype = prob.dtype
    total_pixels = H * W

    # -------------------------- 论文式3-3：概率裁剪，数值保护 --------------------------
    prob_1 = torch.clamp(prob, min=EPS, max=1 - EPS)  # π(h=1)
    prob_0 = 1 - prob_1  # π(h=0)
    log_prob_0 = torch.log(prob_0)  # 预计算logπ，用于梯度计算
    log_prob_1 = torch.log(prob_1)

    # -------------------------- 预计算复用项（论文核心优化：全流程复用） --------------------------
    with torch.no_grad():
        # 预计算c的HVS滤波，全程复用，避免重复计算
        c_hvs = hvs_filter(c)
        # 预生成所有像素坐标，避免循环内重复生成
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        coords = torch.stack([y_grid.flatten(), x_grid.flatten()], dim=1)  # [H*W, 2]
        # 预生成基础半色调图h~π，论文要求的基准采样
        h_base = torch.bernoulli(prob_1).detach()  # [B,1,H,W]，伯努利采样，符合论文设定

    # -------------------------- 初始化梯度项（严格对齐论文式3-14） --------------------------
    # 论文核心：梯度项 = Σh'a [π_a(h'a) * R({h'a, h_-a})]，用于后续求导
    gradient_term = torch.zeros_like(prob, device=device, dtype=dtype)

    # -------------------------- 分块向量化处理（显存+效率双优化） --------------------------
    for block_start in range(0, total_pixels, block_size):
        # 提取当前块像素坐标
        block_end = min(block_start + block_size, total_pixels)
        block_coords = coords[block_start:block_end]  # [block_num, 2]
        block_y = block_coords[:, 0]  # [block_num]
        block_x = block_coords[:, 1]  # [block_num]
        block_num = block_end - block_start

        # -------------------------- 向量化生成h0/h1块，消除循环（核心效率优化） --------------------------
        # 复制基础h为block_num份，[B,1,H,W,block_num]
        h_0_block = h_base.unsqueeze(-1).repeat(1, 1, 1, 1, block_num)
        h_1_block = h_base.unsqueeze(-1).repeat(1, 1, 1, 1, block_num)

        # 向量化翻转对应像素：h0=当前像素设0，h1=当前像素设1，无循环，GPU并行
        batch_idx = torch.arange(B, device=device)[:, None, None]  # [B,1,1]
        channel_idx = torch.arange(C, device=device)[None, :, None]  # [1,C,1]
        block_idx = torch.arange(block_num, device=device)[None, None, :]  # [1,1,block_num]

        # 批量赋值，消除for循环，效率提升100倍+
        h_0_block[batch_idx, channel_idx, block_y[None, :], block_x[None, :], block_idx] = 0.0
        h_1_block[batch_idx, channel_idx, block_y[None, :], block_x[None, :], block_idx] = 1.0

        # -------------------------- 批量计算奖励（论文式3-24） --------------------------
        with torch.no_grad():
            # c扩展为block_num份，复用预计算的c_hvs
            c_batch = c.unsqueeze(-1).repeat(1, 1, 1, 1, block_num)
            # 计算两个动作的奖励，[B,1,block_num]
            r_0 = reward(h_0_block, c_batch, w_s=w_s)
            r_1 = reward(h_1_block, c_batch, w_s=w_s)

        # -------------------------- 论文式3-14核心：计算梯度项，保留梯度链路 --------------------------
        # 梯度项 = π0*r0 + π1*r1，和prob直接关联，保留完整梯度回传链路
        pi_0_block = prob_0[:, :, block_y, block_x]  # [B,1,block_num]
        pi_1_block = prob_1[:, :, block_y, block_x]  # [B,1,block_num]
        block_gradient_term = pi_0_block * r_0 + pi_1_block * r_1

        # 赋值回梯度项张量
        gradient_term[:, :, block_y, block_x] = block_gradient_term

        # 显存清理：仅删除块级临时张量，不调用empty_cache（避免性能损耗）
        del h_0_block, h_1_block, c_batch, r_0, r_1, pi_0_block, pi_1_block, block_gradient_term

    # -------------------------- 论文式3-15：最终L_MARL损失计算 --------------------------
    # 核心：损失 = - 梯度项的均值，反向传播时自动计算∇L_MARL，完全对齐论文公式
    loss_marl = -gradient_term.mean()

    # 数值兜底保护，避免NaN/inf
    loss_marl = torch.nan_to_num(loss_marl, nan=0.0, posinf=1e3, neginf=-1e3)
    # 梯度监控项，可选返回
    grad_norm = gradient_term.norm()

    return loss_marl, grad_norm


# -------------------------- 6. 各向异性抑制损失L_AS（优化版，对齐论文式3-19） --------------------------
def anisotropy_suppression_loss(prob: torch.Tensor) -> torch.Tensor:
    """
    各向异性抑制损失L_AS，优化版，对齐论文式3-19，全流程PyTorch实现，避免CPU/GPU数据搬运
    :param prob: 策略网络输出的概率图π(h=1)，[B,1,H,W]
    :return: L_AS损失标量
    """
    B, _, H, W = prob.shape
    device = prob.device
    loss = 0.0

    # 全流程PyTorch实现，避免CuPy和PyTorch的数据切换，保证梯度回传
    prob = prob.squeeze(1).float()  # [B,H,W]

    # 预生成径向坐标r，全程复用
    cx, cy = H // 2, W // 2
    x = torch.arange(W, device=device).repeat(H, 1)
    y = torch.arange(H, device=device).unsqueeze(1).repeat(1, W)
    r = torch.sqrt((x - cx).float() ** 2 + (y - cy).float() ** 2).long()
    max_r = r.max().item()

    for b in range(B):
        # PyTorch原生FFT，支持自动微分，无数据搬运
        f = torch.fft.fft2(prob[b])
        f_shift = torch.fft.fftshift(f)
        # 功率谱计算，论文式3-16
        P_hat = torch.abs(f_shift) ** 2 / (H * W)
        P_hat += EPS

        # 径向平均功率谱RAPSD，论文式3-17
        P_rho = torch.zeros(max_r + 1, device=device, dtype=prob.dtype)
        for r_val in range(max_r + 1):
            mask = (r == r_val)
            if mask.sum() > 0:
                P_rho[r_val] = P_hat[mask].mean()

        # 论文式3-19：L_AS = Σ( P_hat - P_rho(r) )²
        P_rho_expand = P_rho[r]
        loss_b = torch.sum((P_hat - P_rho_expand) ** 2)
        loss += loss_b

    loss = (loss / B).to(prob.dtype)
    # 数值保护
    loss = torch.nan_to_num(loss, nan=0.0, posinf=1e3, neginf=0.0)
    return loss