import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Union, Optional, List
from functools import lru_cache
from torch import Tensor

# ====================== 全局配置 完全对齐原论文 ======================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE_TYPE = DEVICE.type
# 论文核心常量
HVS_KERNEL_SIZE = 11  # 博士论文指定11x11滤波器尺寸
HVS_SCALE_S = 2000  # 博士论文指定缩放参数S=2000
HVS_HALF_KERNEL = HVS_KERNEL_SIZE // 2
# Näsänen HVS 人眼视觉模型常量 (来自Näsänen 1984原论文)
NASANEN_A = 131.6
NASANEN_B = 0.3188
NASANEN_C = 0.525
NASANEN_D = 3.91
NASANEN_K = 0.85
NASANEN_L = 100.0  # 典型打印场景平均亮度(cd/m²)
# CSSIM 博士论文参数
CSSIM_K1 = 0.01
CSSIM_K2 = 0.03
CSSIM_K = 2.0
DYNAMIC_RANGE_L = 1.0
REWARD_WS = 0.06
# 数值稳定性常量（核心修复：全链路防除零）
EPS = 1e-8
PROB_CLAMP_MIN = 1e-4
PROB_CLAMP_MAX = 1 - 1e-4
DEFAULT_DTYPE = torch.float32


# ====================== 工具函数 ======================
def check_device(tensor: Tensor, func_name: str) -> None:
    if tensor.device.type != DEVICE_TYPE:
        raise RuntimeError(f"{func_name}: 设备类型不匹配，期望{DEVICE_TYPE}，实际{tensor.device.type}")


def safe_contiguous(tensor: Tensor) -> Tensor:
    return tensor.contiguous() if not tensor.is_contiguous() else tensor


# ====================== 预计算缓存 修复Näsänen HVS核核心实现 ======================
@lru_cache(maxsize=8)
def get_precomputed_coords(H: int, W: int) -> Tensor:
    y = torch.arange(H, device=DEVICE)
    x = torch.arange(W, device=DEVICE)
    y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')
    return torch.stack([y_grid.flatten(), x_grid.flatten()], dim=1)


@lru_cache(maxsize=1)
def create_nasanen_hvs_kernel() -> Tensor:
    """
    【修复版】完全实现Näsänen 1984 HVS模型，解决除零警告与核数值异常
    严格对齐博士论文：11x11核尺寸、S=2000缩放参数，保证低通滤波物理意义正确
    """
    kernel_size = HVS_KERNEL_SIZE
    # 1. 计算Näsänen CSF模型核心参数（对比度灵敏度函数）
    S_L = NASANEN_A * (NASANEN_L ** NASANEN_B)
    alpha_L = NASANEN_K / (NASANEN_C * np.log(NASANEN_L) + NASANEN_D)

    # 2. 正确映射空间频率(cycles/degree)到像素域，对齐S=2000缩放参数
    # 博士论文S=2000对应：每度视觉角度包含2000个像素，正确计算像素对应的空间频率
    pixel_per_degree = HVS_SCALE_S
    freq_max = 0.5 * pixel_per_degree  # 奈奎斯特频率
    # 生成归一化频率网格，对应像素域的FFT频率坐标
    freq = np.fft.fftfreq(kernel_size, d=1 / pixel_per_degree)
    fx, fy = np.meshgrid(freq, freq)
    f_radial = np.sqrt(fx ** 2 + fy ** 2)  # 径向空间频率(cycles/degree)

    # 3. 计算CSF频率响应，增加数值稳定性保护
    csf = S_L * np.exp(-alpha_L * f_radial)
    csf = np.clip(csf, a_min=EPS, a_max=None)  # 防止响应为0
    csf = csf / np.max(csf)  # 峰值归一化到1

    # 4. 逆FFT得到空间域点扩散函数(PSF)，修正相位与能量
    psf_complex = np.fft.ifft2(np.fft.ifftshift(csf))
    psf = np.real(psf_complex)  # 取实部，消除数值误差带来的虚部
    psf = np.fft.fftshift(psf)  # 核中心移到矩阵中心

    # 5. 【核心修复】HVS PSF非负约束+防除零归一化
    psf = np.clip(psf, a_min=0.0, a_max=None)  # 人眼PSF非负
    psf_sum = np.sum(psf)  # 用np.sum替代ndarray.sum，解决IDE未解析引用问题
    # 防除零保护，确保总和不会为0
    if psf_sum < EPS:
        psf = np.ones_like(psf)
        psf_sum = np.sum(psf)
    psf = psf / psf_sum  # 归一化，保证核权重和为1，符合低通滤波特性

    # 6. 转换为PyTorch张量，适配卷积格式[out_c, in_c, H, W]
    kernel_tensor = torch.tensor(psf, dtype=DEFAULT_DTYPE, device=DEVICE)
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


# 全局预计算Näsänen HVS核，全程复用
HVS_KERNEL = create_nasanen_hvs_kernel()
HVS_PADDING = HVS_HALF_KERNEL


# ====================== Näsänen HVS低通滤波 核心函数 ======================
def hvs_filter(x: Tensor, padding: int = HVS_PADDING) -> Tensor:
    """
    基于Näsänen HVS模型的人眼视觉低通滤波
    :param padding: 卷积填充值，训练阶段默认HVS_PADDING(5)，指标计算阶段强制传0
    """
    check_device(x, "hvs_filter")
    x = torch.clamp(x, 0.0, 1.0)
    x = safe_contiguous(x)
    # 分组卷积适配多batch/多通道，单通道灰度图完全匹配论文场景
    return F.conv2d(x, HVS_KERNEL, padding=padding, groups=x.shape[1])


# ====================== 逐像素 SSIM 计算（使用Näsänen HVS核）======================
def pixelwise_ssim(
        x: Tensor,
        y: Tensor,
        data_range: float = 1.0,
        K: Tuple[float, float] = (CSSIM_K1, CSSIM_K2),
        win: Optional[Tensor] = None,
) -> Tensor:
    """
    返回与输入相同空间尺寸的 SSIM 图，形状 [B,1,H,W]
    窗口权重使用论文指定的Näsänen HVS核，完全对齐原论文公式
    """
    if x.dim() == 3:
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
    B, C, H, W = x.shape
    assert C == 1, "pixelwise_ssim only supports single channel"
    if win is None:
        win = HVS_KERNEL  # 替换为Näsänen HVS核
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

    # SSIM 公式常数项，增加EPS防除零
    C1 = (K[0] * data_range) ** 2
    C2 = (K[1] * data_range) ** 2
    # 逐像素SSIM图计算
    numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    ssim_map = numerator / (denominator + EPS)
    return ssim_map.clamp(0, 1)


# ====================== CSSIM 核心计算（对比度加权SSIM，论文原公式）======================
def compute_sigma_c(c: Tensor) -> Tensor:
    """预计算连续调图像的对比度图，全程复用Näsänen HVS核"""
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
    完全对齐博士论文式(3-23)：CSSIM = σ_c·SSIM + (1-σ_c)·1
    """
    check_device(c, "cssim")
    check_device(h, "cssim")
    c = torch.clamp(c, 0.0, 1.0)
    h = torch.clamp(h, 0.0, 1.0)
    if sigma_c is None:
        sigma_c = compute_sigma_c(c)
    # 逐像素 SSIM 图
    ssim_map = pixelwise_ssim(c, h, data_range=DYNAMIC_RANGE_L, K=(CSSIM_K1, CSSIM_K2))
    # 对比度加权
    cssim_map = sigma_c * ssim_map + (1 - sigma_c) * 1.0
    cssim_map = torch.clamp(cssim_map, min=EPS, max=1.0)
    return cssim_map.mean(dim=(1, 2, 3))  # [B]


# ====================== 局部MSE奖励变化计算（基于Näsänen HVS）======================
def compute_local_mse_delta(h_sample: Tensor, c_hvs: Tensor, kernel: Tensor) -> Tuple[Tensor, Tensor]:
    """
    高效计算每个像素翻转后的MSE奖励变化 ΔR_mse = -ΔMSE
    完全复用Näsänen HVS核，对齐论文奖励函数定义
    返回两个张量 delta_R0, delta_R1，形状均为 [B,1,H,W]
    """
    B, C, H, W = h_sample.shape
    # 计算当前采样图像的HVS滤波结果
    h_hvs = hvs_filter(h_sample)
    # 差值图
    diff = h_hvs - c_hvs  # [B,1,H,W]
    pad = HVS_HALF_KERNEL
    # 卷积计算局部差值加权和
    conv_diff = F.conv2d(diff, kernel, padding=pad)
    # 卷积计算核平方的局部和
    ones = torch.ones_like(diff)
    kernel_sq = kernel ** 2
    conv_k2 = F.conv2d(ones, kernel_sq, padding=pad)
    factor = 1.0 / (H * W)
    # 原像素值
    h_val = h_sample
    # 两种翻转的像素变化量
    delta_0 = -h_val
    delta_1 = 1 - h_val
    # ΔMSE 公式推导，复用中间结果避免重复计算
    delta_mse_0 = (2 * delta_0 * conv_diff + delta_0 ** 2 * conv_k2) * factor
    delta_mse_1 = (2 * delta_1 * conv_diff + delta_1 ** 2 * conv_k2) * factor
    # 奖励是MSE的负值，ΔR = -ΔMSE
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
    对齐论文奖励函数中CSSIM部分的权重w_s
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


# ====================== 完整LE梯度估计器（论文式3-15）======================
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


# ====================== 各向异性抑制损失（论文式3-19）======================
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
    # 傅里叶变换与功率谱，增加EPS防除零
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
    """基于Näsänen HVS滤波的PSNR计算，对齐论文量化指标"""
    c_hvs = hvs_filter(c, padding=0)
    h_hvs = hvs_filter(h, padding=0)
    mse = F.mse_loss(c_hvs, h_hvs, reduction='none').mean(dim=(1, 2, 3))
    psnr = 10 * torch.log10(1.0 / (mse + EPS))
    return psnr