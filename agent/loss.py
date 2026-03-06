import torch
import torch.nn.functional as F
from typing import Tuple, Union, Optional, List
from functools import lru_cache
from torch import Tensor

# ====================== 全局配置 ======================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE_TYPE = DEVICE.type
# 论文常量（完全对齐原论文）
HVS_KERNEL_SIZE = 11
HVS_SIGMA = 1.5
HVS_HALF_KERNEL = HVS_KERNEL_SIZE // 2
CSSIM_K1 = 0.01
CSSIM_K2 = 0.03
CSSIM_K = 2.0
DYNAMIC_RANGE_L = 1.0
REWARD_WS = 0.06
EPS = 1e-4
PROB_CLAMP_MIN = 1e-4
PROB_CLAMP_MAX = 1 - 1e-4
DEFAULT_DTYPE = torch.float32


# ====================== 工具函数 ======================
def check_device(tensor: Tensor, func_name: str) -> None:
    if tensor.device.type != DEVICE_TYPE:
        raise RuntimeError(f"{func_name}: 设备类型不匹配，期望{DEVICE_TYPE}，实际{tensor.device.type}")

def safe_contiguous(tensor: Tensor) -> Tensor:
    return tensor.contiguous() if not tensor.is_contiguous() else tensor


# ====================== 预计算缓存 ======================
@lru_cache(maxsize=8)
def get_precomputed_coords(H: int, W: int) -> Tensor:
    y = torch.arange(H, device=DEVICE)
    x = torch.arange(W, device=DEVICE)
    y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')
    return torch.stack([y_grid.flatten(), x_grid.flatten()], dim=1)

@lru_cache(maxsize=1)
def create_gaussian_kernel() -> Tensor:
    """全局唯一HVS核，预计算一次，全程复用"""
    coords = torch.arange(HVS_KERNEL_SIZE, device=DEVICE, dtype=DEFAULT_DTYPE) - HVS_HALF_KERNEL
    g_1d = torch.exp(-(coords ** 2) / (2 * HVS_SIGMA ** 2))
    g_1d /= g_1d.sum()
    kernel = torch.outer(g_1d, g_1d)
    kernel /= kernel.sum()
    return kernel[None, None, :, :].contiguous()

@lru_cache(maxsize=16)
def get_radial_coords(H: int, W: int):
    cx, cy = W // 2, H // 2
    y = torch.arange(H, device=DEVICE)
    x = torch.arange(W, device=DEVICE)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    r = torch.sqrt((xx - cx).float() ** 2 + (yy - cy).float() ** 2).long()
    max_r = r.max().item()
    r_count = torch.bincount(r.flatten(), minlength=max_r + 1)
    r_mask = (r_count > 0) & (torch.arange(max_r + 1, device=DEVICE) >= 1)
    r_vals = torch.where(r_mask)[0]
    return r, max_r, r_count, r_mask, r_vals

# 全局预计算常量
HVS_KERNEL = create_gaussian_kernel()
HVS_PADDING = HVS_HALF_KERNEL


# ====================== HVS低通滤波 ======================
def hvs_filter(x: Tensor) -> Tensor:
    check_device(x, "hvs_filter")
    x = torch.clamp(x, 0.0, 1.0)
    x = safe_contiguous(x)
    return F.conv2d(x, HVS_KERNEL, padding=HVS_PADDING, groups=x.shape[1])


# ====================== 逐像素 SSIM 计算（使用HVS核）======================
def pixelwise_ssim(
    x: Tensor,
    y: Tensor,
    data_range: float = 1.0,
    K: Tuple[float, float] = (CSSIM_K1, CSSIM_K2),
    win: Optional[Tensor] = None,
) -> Tensor:
    """
    返回与输入相同空间尺寸的 SSIM 图，形状 [B,1,H,W]
    使用与 HVS 相同的 11x11 高斯核作为窗口权重
    """
    if x.dim() == 3:
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
    B, C, H, W = x.shape
    assert C == 1, "pixelwise_ssim only supports single channel"

    if win is None:
        win = HVS_KERNEL  # [1,1,11,11]

    pad = HVS_HALF_KERNEL
    # 计算局部均值
    mu_x = F.conv2d(x, win, padding=pad)
    mu_y = F.conv2d(y, win, padding=pad)

    # 计算局部方差和协方差
    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x ** 2, win, padding=pad) - mu_x_sq
    sigma_y_sq = F.conv2d(y ** 2, win, padding=pad) - mu_y_sq
    sigma_xy = F.conv2d(x * y, win, padding=pad) - mu_xy

    # 常数
    C1 = (K[0] * data_range) ** 2
    C2 = (K[1] * data_range) ** 2

    # SSIM 图
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2) + EPS)
    return ssim_map.clamp(0, 1)


# ====================== CSSIM 核心计算 ======================
def compute_sigma_c(c: Tensor) -> Tensor:
    """预计算对比度图，全程复用"""
    B, C, H, W = c.shape
    c = torch.clamp(c, 0.0, 1.0)
    c = safe_contiguous(c)
    mu_c = F.conv2d(c, HVS_KERNEL, padding=HVS_PADDING, groups=C)
    c_minus_mu_sq = torch.clamp((c - mu_c) ** 2, min=EPS)
    var_c = F.conv2d(c_minus_mu_sq, HVS_KERNEL, padding=HVS_PADDING, groups=C)
    sigma_c = CSSIM_K * torch.sqrt(var_c)
    sigma_c_max = torch.amax(sigma_c, dim=(1, 2, 3), keepdim=True).clamp(min=EPS)
    sigma_c = sigma_c / sigma_c_max
    return torch.clamp(sigma_c, 0.0, 1.0)

def cssim(c: Tensor, h: Tensor, sigma_c: Optional[Tensor] = None) -> Tensor:
    """
    返回每个样本的CSSIM标量值（逐像素加权后平均）
    """
    check_device(c, "cssim")
    check_device(h, "cssim")
    c = torch.clamp(c, 0.0, 1.0)
    h = torch.clamp(h, 0.0, 1.0)
    if sigma_c is None:
        sigma_c = compute_sigma_c(c)

    # 逐像素 SSIM 图
    ssim_map = pixelwise_ssim(c, h, data_range=DYNAMIC_RANGE_L, K=(CSSIM_K1, CSSIM_K2))

    # 加权
    cssim_map = sigma_c * ssim_map + (1 - sigma_c) * 1.0
    cssim_map = torch.clamp(cssim_map, min=EPS, max=1.0)
    return cssim_map.mean(dim=(1, 2, 3))  # [B]


# ====================== 局部MSE奖励变化计算 ======================
def compute_local_mse_delta(h_sample: Tensor, c_hvs: Tensor, kernel: Tensor) -> Tuple[Tensor, Tensor]:
    """
    高效计算每个像素翻转后的MSE奖励变化 ΔR_mse = -ΔMSE
    返回两个张量 delta_R0, delta_R1，形状均为 [B,1,H,W]
    """
    B, C, H, W = h_sample.shape

    # 计算原始h_hvs
    h_hvs = hvs_filter(h_sample)

    # 差值图
    diff = h_hvs - c_hvs  # [B,1,H,W]

    pad = HVS_HALF_KERNEL
    # conv_diff = conv(diff, kernel)  # 每个位置的 Σ diff_i * K_{a,i}
    conv_diff = F.conv2d(diff, kernel, padding=pad)

    # conv_k2 = conv(ones, kernel^2)  # 每个位置的 Σ K_{a,i}^2
    ones = torch.ones_like(diff)
    kernel_sq = kernel ** 2
    conv_k2 = F.conv2d(ones, kernel_sq, padding=pad)

    factor = 1.0 / (H * W)

    # 原像素值
    h_val = h_sample

    # 两种翻转的变化量
    delta_0 = -h_val
    delta_1 = 1 - h_val

    # ΔMSE = (2*Δh*conv_diff + Δh^2*conv_k2) * factor
    delta_mse_0 = (2 * delta_0 * conv_diff + delta_0**2 * conv_k2) * factor
    delta_mse_1 = (2 * delta_1 * conv_diff + delta_1**2 * conv_k2) * factor

    # ΔR_mse = -ΔMSE
    delta_R0 = -delta_mse_0
    delta_R1 = -delta_mse_1

    return delta_R0.detach(), delta_R1.detach()


# ====================== 局部CSSIM奖励变化计算 ======================
def compute_local_cssim_delta(
    c: Tensor,
    h_sample: Tensor,
    sigma_c: Tensor,
    w_s: float
) -> Tuple[Tensor, Tensor]:
    """
    使用自动微分计算每个像素翻转对CSSIM奖励的影响（一阶近似）
    返回 delta_C0, delta_C1，形状 [B,1,H,W]
    """
    # 创建可微副本用于梯度计算
    h_grad = h_sample.detach().clone().requires_grad_(True)

    # 计算当前CSSIM值（标量，每个样本一个）
    cssim_val = cssim(c, h_grad, sigma_c)  # [B]

    # 求CSSIM对h的梯度（每个像素的导数）
    grad_h = torch.autograd.grad(cssim_val.sum(), h_grad, retain_graph=False)[0]  # [B,1,H,W]

    # 像素翻转的变化量
    delta_h0 = -h_sample
    delta_h1 = 1 - h_sample

    # 一阶近似 ΔCSSIM ≈ grad_h * Δh
    delta_cssim0 = grad_h * delta_h0
    delta_cssim1 = grad_h * delta_h1

    # 奖励中CSSIM部分的变化量 = w_s * ΔCSSIM
    delta_C0 = w_s * delta_cssim0
    delta_C1 = w_s * delta_cssim1

    return delta_C0.detach(), delta_C1.detach()


# ====================== 完整LE梯度估计器 ======================
def le_gradient_estimator(
        c: torch.Tensor,
        prob: torch.Tensor,
        w_s: float = REWARD_WS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    完全对齐原论文LE梯度估计器公式 (3-15)
    包括MSE和CSSIM两部分奖励变化，返回 (loss_marl, grad_norm)
    """
    B, C, H, W = prob.shape
    check_device(c, "le_gradient_estimator")
    check_device(prob, "le_gradient_estimator")

    c = c.to(dtype=DEFAULT_DTYPE, non_blocking=True)
    prob = torch.clamp(prob, PROB_CLAMP_MIN, PROB_CLAMP_MAX)

    # 1. 采样二值图像 h ~ Bernoulli(prob)
    h_sample = torch.bernoulli(prob)  # [B,1,H,W]

    # 2. 预计算HVS滤波结果和对比度图（用于CSSIM）
    c_hvs = hvs_filter(c)
    sigma_c = compute_sigma_c(c)

    # 3. 计算MSE部分的奖励变化
    delta_Rmse0, delta_Rmse1 = compute_local_mse_delta(h_sample, c_hvs, HVS_KERNEL)

    # 4. 计算CSSIM部分的奖励变化（利用梯度近似）
    delta_C0, delta_C1 = compute_local_cssim_delta(c, h_sample, sigma_c, w_s)

    # 5. 总奖励变化
    delta_R0 = delta_Rmse0 + delta_C0
    delta_R1 = delta_Rmse1 + delta_C1

    # 6. 策略概率
    prob_1 = prob  # π(1)
    prob_0 = 1 - prob_1  # π(0)

    # 7. 计算MARL损失（公式3-15的负号）
    loss_per_pixel = prob_0 * delta_R0 + prob_1 * delta_R1  # [B,1,H,W]
    loss_marl = -loss_per_pixel.mean()

    # 8. 梯度范数（用于日志，近似估计）
    grad_norm = (delta_R0.norm() + delta_R1.norm()) / (B * H * W)

    return loss_marl, grad_norm


# ====================== 各向异性抑制损失 ======================
def anisotropy_suppression_loss(prob: Tensor) -> Tensor:
    """
    优化后损失函数，向量化实现，完全对齐公式 (3-19)
    注意：此函数应在恒定灰度图对应的概率输出上调用
    """
    check_device(prob, "anisotropy_suppression_loss")
    B, C, H, W = prob.shape
    prob_sq = prob.squeeze(1).to(DEFAULT_DTYPE)  # [B,H,W]
    prob_sq = torch.clamp(prob_sq, min=PROB_CLAMP_MIN, max=PROB_CLAMP_MAX)

    # 预计算径向坐标
    r, max_r, r_count, r_mask, r_vals = get_radial_coords(H, W)
    r_flat = r.flatten()

    # 傅里叶变换与功率谱
    fft = torch.fft.fft2(prob_sq)
    fft_shift = torch.fft.fftshift(fft)
    P_hat = torch.abs(fft_shift) ** 2 / (H * W) + EPS  # [B,H,W]
    P_flat = P_hat.reshape(B, -1)  # [B, H*W]

    # 径向平均功率谱
    P_rho = torch.zeros(B, max_r + 1, device=DEVICE, dtype=DEFAULT_DTYPE)
    P_rho.scatter_add_(1, r_flat.unsqueeze(0).expand(B, -1), P_flat)
    r_count_safe = r_count.clamp(min=1)
    P_rho = P_rho / r_count_safe.unsqueeze(0)  # [B, max_r+1]

    # 各向异性损失
    P_rho_expand = P_rho[:, r_flat].reshape(B, H, W)  # [B,H,W]
    non_dc = (r >= 1).unsqueeze(0).expand_as(P_hat)
    loss = ((P_hat - P_rho_expand) ** 2 * non_dc).mean(dim=(1, 2)).mean()

    return torch.clamp(loss, min=1e-6).to(DEFAULT_DTYPE)


# ====================== 验证指标：HVS-PSNR ======================
def calculate_hvs_psnr(c: Tensor, h: Tensor) -> Tensor:
    c_hvs = hvs_filter(c)
    h_hvs = hvs_filter(h)
    mse = F.mse_loss(c_hvs, h_hvs, reduction='none').mean(dim=(1, 2, 3))
    psnr = 10 * torch.log10(1.0 / (mse + EPS))
    return psnr