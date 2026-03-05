import torch
import torch.nn.functional as F
from typing import Tuple
from functools import lru_cache
from profilehooks import profile
from torch import Tensor
from pytorch_msssim import ssim

# ====================== 全局配置 ======================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE_TYPE = DEVICE.type

# 论文常量
HVS_KERNEL_SIZE = 11
HVS_SIGMA = 1.5
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
        raise RuntimeError(f"{func_name}: 设备类型不匹配")

def safe_contiguous(tensor: Tensor) -> Tensor:
    return tensor.contiguous() if not tensor.is_contiguous() else tensor

# ====================== 坐标网格缓存 ======================
@lru_cache(maxsize=8)
def get_precomputed_coords(H: int, W: int) -> Tensor:
    y = torch.arange(H, device=DEVICE)
    x = torch.arange(W, device=DEVICE)
    y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')
    return torch.stack([y_grid.flatten(), x_grid.flatten()], dim=1)

# ====================== 高斯核生成 ======================
def create_gaussian_kernel(size: int, sigma: float) -> Tensor:
    coords = torch.arange(size, device=DEVICE, dtype=torch.float32) - size // 2
    g_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g_1d /= g_1d.sum()
    kernel = torch.outer(g_1d, g_1d)
    kernel /= kernel.sum()
    return kernel[None, None, :, :].contiguous()

HVS_KERNEL = create_gaussian_kernel(HVS_KERNEL_SIZE, HVS_SIGMA) if DEVICE_TYPE == 'cuda' else None
HVS_PADDING = HVS_KERNEL_SIZE // 2

# ====================== HVS低通滤波 ======================
def hvs_filter(x: Tensor) -> Tensor:
    check_device(x, "hvs_filter")
    x = torch.clamp(x, 0.0, 1.0)
    x = safe_contiguous(x)
    if HVS_KERNEL is None:
        raise RuntimeError("HVS核未初始化")
    return F.conv2d(x, HVS_KERNEL, padding=HVS_PADDING, groups=x.shape[1])

def batch_hvs_filter(h_batch: Tensor) -> Tensor:
    B, C, H, W, N = h_batch.shape
    h_reshaped = h_batch.permute(0, 4, 1, 2, 3).reshape(-1, C, H, W)
    h_reshaped = safe_contiguous(h_reshaped)
    h_hvs = hvs_filter(h_reshaped)
    return h_hvs.reshape(B, N, C, H, W).permute(0, 2, 3, 4, 1)  # [B, C, H, W, N]

# ====================== 对比度σ_c计算 ======================
def compute_sigma_c(c: Tensor) -> Tensor:
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

# ====================== CSSIM计算 ======================
def cssim(c: Tensor, h: Tensor) -> Tensor:
    check_device(c, "cssim")
    check_device(h, "cssim")
    c = torch.clamp(c, 0.0, 1.0)
    h = torch.clamp(h, 0.0, 1.0)
    sigma_c = compute_sigma_c(c)
    ssim_map = ssim(
        X=c, Y=h, data_range=DYNAMIC_RANGE_L, size_average=False,
        win_size=HVS_KERNEL_SIZE, win_sigma=HVS_SIGMA,
        K=(CSSIM_K1, CSSIM_K2), nonnegative_ssim=False
    )[0]
    cssim_map = sigma_c * ssim_map + (1 - sigma_c) * 1.0
    cssim_map = torch.clamp(cssim_map, min=EPS, max=1.0)
    return cssim_map.mean(dim=(1, 2, 3))

def batch_cssim(h_batch: Tensor, c_batch: Tensor) -> Tensor:
    B, C, H, W, N = h_batch.shape
    h_reshaped = h_batch.permute(0, 4, 1, 2, 3).reshape(-1, C, H, W)
    c_reshaped = c_batch.permute(0, 4, 1, 2, 3).reshape(-1, C, H, W)
    scores = cssim(h_reshaped, c_reshaped)  # [B*N]
    return scores.reshape(B, N).unsqueeze(1)  # [B, 1, N]

# ====================== 奖励函数（修复批处理分支）======================
def reward(h: Tensor, c: Tensor, w_s: float = REWARD_WS) -> Tensor:
    with torch.no_grad():
        check_device(h, "reward")
        check_device(c, "reward")
        is_batch = h.dim() == 5
        h = torch.clamp(h, 0.0, 1.0)
        c = torch.clamp(c, 0.0, 1.0)

        if is_batch:
            B, C, H, W, N = h.shape
            h_hvs = batch_hvs_filter(h)          # [B, C, H, W, N]
            c_hvs = batch_hvs_filter(c)
            # 显式 reshape 计算每个样本的 MSE
            h_reshaped = h_hvs.permute(0, 4, 1, 2, 3).reshape(-1, C, H, W)  # [B*N, C, H, W]
            c_reshaped = c_hvs.permute(0, 4, 1, 2, 3).reshape(-1, C, H, W)
            mse_per_sample = F.mse_loss(h_reshaped, c_reshaped, reduction='none').mean(dim=(1, 2, 3))  # [B*N]
            mse = mse_per_sample.reshape(B, N)   # [B, N]
            cssim_score = batch_cssim(h, c)      # [B, 1, N]
            r = -mse.unsqueeze(1) + w_s * cssim_score  # [B, 1, N]
        else:
            h_hvs = hvs_filter(h)
            c_hvs = hvs_filter(c)
            mse = F.mse_loss(h_hvs, c_hvs, reduction='none').mean(dim=(1, 2, 3))
            cssim_score = cssim(h, c).unsqueeze(1)  # [B, 1]
            r = -mse.unsqueeze(1) + w_s * cssim_score  # [B, 1]

        return torch.clamp(r, min=-10.0, max=1.0)

# ====================== LE梯度估计器（修复赋值）======================
@profile
def le_gradient_estimator(
        c: torch.Tensor,
        prob: torch.Tensor,
        w_s: float = 0.06,
        block_size: int = 16,
        max_grad_norm: float = 10.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, C, H, W = prob.shape
    device = prob.device
    total_pixels = H * W

    c = c.to(dtype=DEFAULT_DTYPE, non_blocking=True)
    prob_1 = torch.clamp(prob, min=PROB_CLAMP_MIN, max=PROB_CLAMP_MAX)

    with torch.no_grad():
        coords = get_precomputed_coords(H, W)
        h_base = torch.bernoulli(prob_1).detach().contiguous()
        c_expanded = c.unsqueeze(-1)  # [B, C, H, W, 1]

    delta_r = torch.zeros_like(prob_1, dtype=DEFAULT_DTYPE, device=device)

    for block_start in range(0, total_pixels, block_size):
        block_end = min(block_start + block_size, total_pixels)
        block_coords = coords[block_start:block_end]
        block_y, block_x = block_coords[:, 0], block_coords[:, 1]
        block_num = block_end - block_start

        # 构建两个翻转后的halftone块
        h_0 = h_base.unsqueeze(-1).expand(-1, -1, -1, -1, block_num).clone()
        h_1 = h_base.unsqueeze(-1).expand(-1, -1, -1, -1, block_num).clone()
        h_0 = safe_contiguous(h_0)
        h_1 = safe_contiguous(h_1)

        batch_idx = torch.arange(B, device=device)[:, None]
        chan_idx = torch.arange(C, device=device)[None, :]
        h_0[batch_idx, chan_idx, block_y, block_x, :] = 0.0
        h_1[batch_idx, chan_idx, block_y, block_x, :] = 1.0

        c_batch = c_expanded.expand(-1, -1, -1, -1, block_num)
        c_batch = safe_contiguous(c_batch)

        r_0 = reward(h_0, c_batch, w_s=w_s)  # [B, 1, block_num]
        r_1 = reward(h_1, c_batch, w_s=w_s)  # [B, 1, block_num]
        block_delta = r_1 - r_0               # [B, 1, block_num]

        # 赋值，让广播处理通道维
        delta_r[:, :, block_y, block_x] = block_delta  # [B,1,block_num] -> [B,C,block_num]

    loss_marl = -(delta_r * prob_1).mean()
    loss_marl = torch.nan_to_num(loss_marl, nan=0.0, posinf=1e3, neginf=-1e3)
    grad_norm = torch.nan_to_num(delta_r.norm(), nan=0.0)
    return loss_marl, grad_norm

# ====================== 各向异性抑制损失（保持不变）======================
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

def anisotropy_suppression_loss(prob: Tensor) -> Tensor:
    check_device(prob, "anisotropy_suppression_loss")
    B, C, H, W = prob.shape
    prob_sq = prob.squeeze(1).to(DEFAULT_DTYPE)  # [B,H,W]
    prob_sq = torch.clamp(prob_sq, min=PROB_CLAMP_MIN, max=PROB_CLAMP_MAX)

    r, max_r, _, _, r_vals = get_radial_coords(H, W)

    fft = torch.fft.fft2(prob_sq)
    fft_shift = torch.fft.fftshift(fft)
    P_hat = torch.abs(fft_shift) ** 2 / (H * W) + EPS

    r_flat = r.flatten()
    P_flat = P_hat.reshape(B, -1)
    P_rho = torch.zeros(B, max_r + 1, device=DEVICE, dtype=torch.float32)
    for rv in r_vals:
        mask = (r_flat == rv)
        P_rho[:, rv] = P_flat[:, mask].mean(dim=1)

    P_rho_expand = P_rho[:, r]  # [B,H,W]
    non_dc = (r >= 1).unsqueeze(0).expand_as(P_hat)
    loss = ((P_hat - P_rho_expand) ** 2 * non_dc).mean(dim=(1, 2)).mean()
    return torch.clamp(loss, min=1e-6).to(DEFAULT_DTYPE)

# ====================== HVS-PSNR ======================
def calculate_hvs_psnr(c: Tensor, h: Tensor) -> Tensor:
    check_device(c, "calculate_hvs_psnr")
    check_device(h, "calculate_hvs_psnr")
    c_hvs = hvs_filter(c)
    h_hvs = hvs_filter(h)
    mse = F.mse_loss(c_hvs, h_hvs)
    return 10 * torch.log10(1.0 / (mse + EPS))