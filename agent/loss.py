import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Union, Optional, List
from functools import lru_cache
from torch import Tensor

# ====================== 全局配置 100%对齐原论文 ======================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE_TYPE = DEVICE.type

# 论文核心常量
HVS_KERNEL_SIZE = 11
HVS_SCALE_S = 2000
HVS_HALF_KERNEL = HVS_KERNEL_SIZE // 2

# Näsänen HVS 人眼视觉模型常量
NASANEN_A = 131.6
NASANEN_B = 0.3188
NASANEN_C = 0.525
NASANEN_D = 3.91
NASANEN_K = 0.85
NASANEN_L = 100.0

# CSSIM 博士论文参数
CSSIM_K1 = 0.01
CSSIM_K2 = 0.03
CSSIM_K = 2.0
DYNAMIC_RANGE_L = 1.0
REWARD_WS = 0.06
CSSIM_WIN_SIGMA = 1.5

# 数值稳定性常量
EPS = 1e-12
PROB_CLAMP_MIN = 1e-4
PROB_CLAMP_MAX = 1 - 1e-4
DEFAULT_DTYPE = torch.float32


# ====================== 工具函数 ======================
def check_device(tensor: Tensor, func_name: str) -> None:
    if tensor.device.type != DEVICE_TYPE:
        raise RuntimeError(f"{func_name}: 设备不匹配，期望{DEVICE_TYPE}，实际{tensor.device.type}")


def safe_contiguous(tensor: Tensor) -> Tensor:
    return tensor.contiguous() if not tensor.is_contiguous() else tensor


# ====================== 预计算核缓存 ======================
@lru_cache(maxsize=1)
def create_nasanen_hvs_kernel() -> Tensor:
    """
    严格实现Näsänen 1984 HVS模型，空间域直接生成
    对齐博士论文：11x11核、S=2000缩放参数
    """
    kernel_size = HVS_KERNEL_SIZE
    half_size = HVS_HALF_KERNEL
    pixel_per_degree = HVS_SCALE_S

    x = np.arange(-half_size, half_size + 1, dtype=np.float64) / pixel_per_degree
    y = np.arange(-half_size, half_size + 1, dtype=np.float64) / pixel_per_degree
    xx, yy = np.meshgrid(x, y)

    S_L = NASANEN_A * (NASANEN_L ** NASANEN_B)
    alpha_L = NASANEN_K / (NASANEN_C * np.log(NASANEN_L) + NASANEN_D)

    f_cutoff = np.log(S_L / 1.0) / alpha_L
    sigma_pixel = pixel_per_degree / (2 * np.pi * f_cutoff)
    psf = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma_pixel ** 2))

    psf = np.clip(psf, a_min=0.0, a_max=None)
    psf_sum = np.sum(psf)
    if psf_sum < float(EPS):
        psf = np.ones_like(psf)
        psf_sum = np.sum(psf)
    psf = psf / psf_sum

    kernel_tensor = torch.tensor(psf, dtype=DEFAULT_DTYPE, device=DEVICE)
    return kernel_tensor[None, None, :, :].contiguous()


@lru_cache(maxsize=1)
def create_cssim_gaussian_kernel() -> Tensor:
    """CSSIM σ_c计算专用11x11高斯核，标准差1.5"""
    kernel_size = HVS_KERNEL_SIZE
    sigma = CSSIM_WIN_SIGMA
    half_size = HVS_HALF_KERNEL

    x = np.arange(-half_size, half_size + 1, dtype=np.float64)
    y = np.arange(-half_size, half_size + 1, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    gauss = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    gauss = gauss / np.sum(gauss)

    kernel_tensor = torch.tensor(gauss, dtype=DEFAULT_DTYPE, device=DEVICE)
    return kernel_tensor[None, None, :, :].contiguous()


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


# 全局预计算核
HVS_KERNEL = create_nasanen_hvs_kernel()
CSSIM_GAUSS_KERNEL = create_cssim_gaussian_kernel()
HVS_PADDING = HVS_HALF_KERNEL


# ====================== Näsänen HVS低通滤波 ======================
def hvs_filter(x: Tensor, padding: int = HVS_PADDING) -> Tensor:
    """基于Näsänen HVS模型的人眼视觉低通滤波"""
    check_device(x, "hvs_filter")
    x = torch.clamp(x, 0.0, 1.0)
    x = safe_contiguous(x)
    return F.conv2d(x, HVS_KERNEL, padding=padding, groups=x.shape[1])


# ====================== 逐像素 SSIM ======================
def pixelwise_ssim(
        x: Tensor,
        y: Tensor,
        data_range: float = 1.0,
        K: Tuple[float, float] = (CSSIM_K1, CSSIM_K2),
        win: Optional[Tensor] = None,
) -> Tensor:
    """返回 SSIM 图 [B,1,H,W]"""
    if x.dim() == 3:
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
    B, C, H, W = x.shape
    assert C == 1, "pixelwise_ssim only supports single channel"

    if win is None:
        win = CSSIM_GAUSS_KERNEL
    pad = HVS_HALF_KERNEL

    mu_x = F.conv2d(x, win, padding=pad, groups=C)
    mu_y = F.conv2d(y, win, padding=pad, groups=C)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y
    sigma_x_sq = F.conv2d(x ** 2, win, padding=pad, groups=C) - mu_x_sq
    sigma_y_sq = F.conv2d(y ** 2, win, padding=pad, groups=C) - mu_y_sq
    sigma_xy = F.conv2d(x * y, win, padding=pad, groups=C) - mu_xy

    C1 = (K[0] * data_range) ** 2
    C2 = (K[1] * data_range) ** 2

    numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    ssim_map = numerator / (denominator + EPS)
    return ssim_map.clamp(0, 1)


# ====================== CSSIM（论文式3-23）======================
def compute_sigma_c(c: Tensor) -> Tensor:
    """对齐论文式3-22"""
    B, C, H, W = c.shape
    c = torch.clamp(c, 0.0, 1.0)
    c = safe_contiguous(c)

    mu_c = F.conv2d(c, CSSIM_GAUSS_KERNEL, padding=HVS_PADDING, groups=C)
    c_minus_mu_sq = torch.clamp((c - mu_c) ** 2, min=EPS)
    var_c = F.conv2d(c_minus_mu_sq, CSSIM_GAUSS_KERNEL, padding=HVS_PADDING, groups=C)
    sigma_c = CSSIM_K * torch.sqrt(var_c)
    sigma_c_max = torch.amax(sigma_c, dim=(1, 2, 3), keepdim=True).clamp(min=EPS)
    sigma_c = sigma_c / sigma_c_max
    return torch.clamp(sigma_c, 0.0, 1.0)


def cssim(c: Tensor, h: Tensor, sigma_c: Optional[Tensor] = None) -> Tensor:
    """CSSIM = σ_c·SSIM + (1-σ_c)·1, 返回 [B]"""
    check_device(c, "cssim")
    check_device(h, "cssim")
    c = torch.clamp(c, 0.0, 1.0)
    h = torch.clamp(h, 0.0, 1.0)

    if sigma_c is None:
        sigma_c = compute_sigma_c(c)

    ssim_map = pixelwise_ssim(c, h, data_range=DYNAMIC_RANGE_L)
    cssim_map = sigma_c * ssim_map + (1 - sigma_c) * 1.0
    cssim_map = torch.clamp(cssim_map, min=EPS, max=1.0)
    return cssim_map.mean(dim=(1, 2, 3))  # [B]


# ====================== LE梯度估计器（论文式3-14/3-15）======================
def le_gradient_estimator(
        c: torch.Tensor,
        prob: torch.Tensor,
        w_s: float = REWARD_WS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    论文式(3-15) LE梯度估计器

    核心思路：
    论文公式 ∇_θ L = -E[Σ_a Σ_{h_a'} ∇_θ π_a(h_a') R({h_a',h_{-a}})]

    等价REINFORCE形式：
    L = -E_h[ log π(h) * ( R(h) - baseline ) ]

    baseline用多采样估计减方差。

    关键：ΔR不除以N，保持O(1)量级的梯度信号。
    论文中的奖励R是对"整张图"的全局评价，
    但LE估计器关心的是每个像素翻转对R的影响——这个影响
    不应该被N稀释，因为我们是在做逐像素的策略梯度更新。
    """
    B, C, H, W = prob.shape
    N = H * W
    check_device(c, "le_gradient_estimator")
    check_device(prob, "le_gradient_estimator")
    c = c.to(dtype=DEFAULT_DTYPE, non_blocking=True)
    prob = prob.to(dtype=DEFAULT_DTYPE, non_blocking=True)
    prob = torch.clamp(prob, PROB_CLAMP_MIN, PROB_CLAMP_MAX)

    # 1. 采样二值图像
    h_sample = torch.bernoulli(prob).detach()

    # 2. 预计算
    sigma_c = compute_sigma_c(c)
    c_hvs = hvs_filter(c, padding=HVS_PADDING)
    h_hvs = hvs_filter(h_sample, padding=HVS_PADDING)
    diff = h_hvs - c_hvs  # [B,1,H,W]

    # 3. MSE翻转变化量——不除以N，保持O(1)量级
    pad = HVS_HALF_KERNEL
    conv_diff = F.conv2d(diff, HVS_KERNEL, padding=pad, groups=C)
    kernel_sq = HVS_KERNEL ** 2
    conv_k2 = F.conv2d(torch.ones_like(diff), kernel_sq, padding=pad, groups=C)

    h_val = h_sample
    delta_0 = -h_val
    delta_1 = 1.0 - h_val

    # ΔR_mse(a) = -(2*δ*conv_diff + δ²*conv_k2)  【不除以N】
    delta_Rmse_0 = -(2.0 * delta_0 * conv_diff + delta_0 ** 2 * conv_k2)
    delta_Rmse_1 = -(2.0 * delta_1 * conv_diff + delta_1 ** 2 * conv_k2)

    # 4. CSSIM翻转变化量（自动微分一阶近似）
    # cssim()内部有mean(1/N)，所以grad_cssim自带1/N量级
    # 乘回N使其与MSE部分对齐到O(1)量级
    h_grad = h_sample.detach().clone().requires_grad_(True)
    cssim_val = cssim(c, h_grad, sigma_c)
    grad_cssim = torch.autograd.grad(cssim_val.sum(), h_grad, retain_graph=False)[0]
    # grad_cssim 自带 1/N，乘回 N 使量级对齐
    grad_cssim = grad_cssim * N

    delta_Rcssim_0 = w_s * grad_cssim * delta_0
    delta_Rcssim_1 = w_s * grad_cssim * delta_1

    # 5. 总ΔR
    delta_R0 = (delta_Rmse_0 + delta_Rcssim_0).detach()
    delta_R1 = (delta_Rmse_1 + delta_Rcssim_1).detach()

    # 6. 策略概率
    prob_1 = prob
    prob_0 = 1.0 - prob_1

    # 7. 计算每像素的期望优势（作为advantage）
    # baseline = E_{h_a'}[ΔR] = π_0*ΔR_0 + π_1*ΔR_1
    baseline = (prob_0 * delta_R0 + prob_1 * delta_R1).detach()
    A0 = (delta_R0 - baseline).detach()
    A1 = (delta_R1 - baseline).detach()

    # 8. REINFORCE: L = -E[ log π(h_a) * A(h_a) ]
    log_prob_1 = torch.log(prob_1 + EPS)
    log_prob_0 = torch.log(prob_0 + EPS)

    log_pi_selected = h_sample * log_prob_1 + (1.0 - h_sample) * log_prob_0
    advantage_selected = h_sample * A1 + (1.0 - h_sample) * A0

    loss_marl = -(log_pi_selected * advantage_selected).mean()

    # 9. 梯度监控
    grad_norm = advantage_selected.abs().mean().detach()

    return loss_marl, grad_norm


# ====================== 各向异性抑制损失（论文式3-19）======================
def anisotropy_suppression_loss(prob: Tensor) -> Tensor:
    """向量化实现，对齐论文式 (3-19)"""
    check_device(prob, "anisotropy_suppression_loss")
    B, C, H, W = prob.shape
    prob_sq = prob.squeeze(1).to(DEFAULT_DTYPE)
    prob_sq = torch.clamp(prob_sq, min=PROB_CLAMP_MIN, max=PROB_CLAMP_MAX)

    r, max_r, r_count, r_mask, r_vals = get_radial_coords(H, W)
    r_flat = r.flatten()

    fft = torch.fft.fft2(prob_sq)
    fft_shift = torch.fft.fftshift(fft)
    P_hat = torch.abs(fft_shift) ** 2 / (H * W) + EPS
    P_flat = P_hat.reshape(B, -1)

    P_rho = torch.zeros(B, max_r + 1, device=DEVICE, dtype=DEFAULT_DTYPE)
    P_rho.scatter_add_(1, r_flat.unsqueeze(0).expand(B, -1), P_flat)
    r_count_safe = r_count.clamp(min=1)
    P_rho = P_rho / r_count_safe.unsqueeze(0)

    P_rho_expand = P_rho[:, r_flat].reshape(B, H, W)
    non_dc = (r >= 1).unsqueeze(0).expand_as(P_hat)
    loss = ((P_hat - P_rho_expand) ** 2 * non_dc).mean(dim=(1, 2)).mean()

    return torch.clamp(loss, min=1e-6).to(DEFAULT_DTYPE)


# ====================== 验证指标：HVS-PSNR ======================
def calculate_hvs_psnr(c: Tensor, h: Tensor) -> Tensor:
    """基于Näsänen HVS滤波的PSNR计算"""
    c_hvs = hvs_filter(c, padding=0)
    h_hvs = hvs_filter(h, padding=0)
    mse = F.mse_loss(c_hvs, h_hvs, reduction='none').mean(dim=(1, 2, 3))
    psnr = 10 * torch.log10(1.0 / (mse + EPS))
    return psnr