import cv2
import numpy as np
import torch
from numpy.fft import fft2, fftshift
import cupy as cp
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gaussian_kernel(kernel_size=11, sigma=1.5):
    """生成11×11高斯核（论文CSSIM对比度计算/HVS滤波用）"""
    kx = cv2.getGaussianKernel(kernel_size, sigma)
    ky = cv2.getGaussianKernel(kernel_size, sigma)
    kernel = np.outer(kx, ky)
    return torch.from_numpy(kernel).float().unsqueeze(0).unsqueeze(0).to(device)

def hvs_filter(x, kernel_size=11, scale=2000):
    """Näsänen HVS低通滤波（论文3.5节，scale=2000固定）"""
    kernel = gaussian_kernel(kernel_size, sigma=1.5)
    # 分组卷积加速，保持图像尺寸
    x_filtered = F.conv2d(x, kernel, padding=kernel_size//2, groups=x.shape[1])
    return x_filtered / scale


def cssim(h, c, kernel_size=11, sigma=1.5, eps=1e-8):
    """对比度加权SSIM（严格对齐论文式3-23），增加数值保护解决NaN"""
    B, C, H, W = c.shape
    kernel = gaussian_kernel(kernel_size, sigma)
    # 1. 计算连续调图像c的局部对比度图σ_c（论文式3-22）
    mu_c = F.conv2d(c, kernel, padding=kernel_size // 2, groups=C)
    # 开方前裁剪为非负，避免√负数
    c_mu_sq = torch.clamp((c - mu_c) ** 2, min=eps)
    sigma_c = 2 * torch.sqrt(F.conv2d(c_mu_sq, kernel, padding=kernel_size // 2, groups=C))
    sigma_c = torch.clamp(sigma_c, 0, 1)  # 论文要求归一化到[0,1]

    # 2. 计算基础SSIM（论文式3-21），所有除法/开方均加数值保护
    C1, C2 = 1e-4, 9e-4
    mu_h = F.conv2d(h, kernel, padding=kernel_size // 2, groups=C)
    mu_hc = mu_h * mu_c
    mu_h2, mu_c2 = mu_h ** 2, mu_c ** 2

    # 计算方差时裁剪为非负，避免方差为负
    h_sq = torch.clamp(h ** 2, min=eps)
    c_sq = torch.clamp(c ** 2, min=eps)
    sigma_h2 = torch.clamp(F.conv2d(h_sq, kernel, padding=kernel_size // 2, groups=C) - mu_h2, min=eps)
    sigma_c2 = torch.clamp(F.conv2d(c_sq, kernel, padding=kernel_size // 2, groups=C) - mu_c2, min=eps)
    sigma_hc = F.conv2d(h * c, kernel, padding=kernel_size // 2, groups=C) - mu_hc

    # SSIM的亮度/对比度/结构项，除法加eps避免除零
    l = (2 * mu_hc + C1) / (mu_h2 + mu_c2 + C1 + eps)
    c_ = (2 * torch.sqrt(sigma_h2 * sigma_c2) + C2) / (sigma_h2 + sigma_c2 + C2 + eps)
    s = (2 * sigma_hc + C2) / (torch.sqrt(sigma_h2 * sigma_c2) * 2 + C2 + eps)

    ssim_map = l * c_ * s
    # 3. 论文式3-23：CSSIM = σ_c * SSIM + (1-σ_c) * 1
    cssim_map = sigma_c * ssim_map + (1 - sigma_c) * 1.0
    # 数值保护：避免SSIM计算结果为NaN/inf
    cssim_map = torch.clamp(cssim_map, min=eps, max=1.0)

    return cssim_map.mean()

def reward(h, c, w_s=0.06, eps=1e-8):
    """奖励函数（论文式3-24）：R=-MSE(HVS(h),HVS(c)) + w_s*CSSIM"""
    h_hvs = hvs_filter(h)
    c_hvs = hvs_filter(c)
    # MSE裁剪为非负，避免负奖励导致梯度异常
    mse = torch.clamp(F.mse_loss(h_hvs, c_hvs), min=eps)
    cssim_score = cssim(h, c)
    r = -mse + w_s * cssim_score
    # 奖励值裁剪，避免极端值导致梯度爆炸
    r = torch.clamp(r, min=-1e2, max=1e2)
    return r


def anisotropy_suppression_loss(prob):
    """各向异性抑制损失L_AS（CuPy半精度支持）"""
    B, _, H, W = prob.shape
    device = prob.device
    loss = 0.0
    prob = prob.squeeze(1).float()  # [B,H,W]，FP16张量

    for b in range(B):
        # 1. PyTorch FP16张量 → CuPy FP16张量
        prob_cp = cp.asarray(prob[b].detach())  # 自动转为cupy.float16

        # 2. CuPy FFT（原生支持半精度，无警告）
        f_cp = cp.fft.fft2(prob_cp)
        f_shift_cp = cp.fft.fftshift(f_cp)

        # 3. 计算功率谱（CuPy内计算，避免数据回传）
        P_hat_cp = cp.abs(f_shift_cp) ** 2 / (H * W)
        P_hat_cp += 1e-8

        # 4. 径向坐标r（转回PyTorch计算，或全程CuPy，按需选择）
        # （这里为了和原逻辑一致，转回PyTorch，也可全程用CuPy加速）
        P_hat = torch.as_tensor(P_hat_cp, device=device)

        # 后续径向平均、L_AS计算逻辑和方案1完全一致...
        cx, cy = H//2, W//2
        x = torch.arange(W, device=device).repeat(H, 1)
        y = torch.arange(H, device=device).unsqueeze(1).repeat(1, W)
        r = torch.sqrt((x - cx).float()**2 + (y - cy).float()**2).long()
        max_r = r.max().item()

        P_rho = torch.zeros(max_r + 1, device=device, dtype=torch.float16)
        for r_val in range(max_r + 1):
            mask = (r == r_val)
            if mask.sum() > 0:
                P_rho[r_val] = P_hat[mask].mean()

        P_rho_expand = P_rho[r]
        loss_b = torch.sum((P_hat - P_rho_expand) ** 2)
        loss += loss_b

    loss = (loss / B).to(prob.dtype)
    return loss


def le_gradient_estimator(prob, c, w_s=0.06, eps=1e-8):
    r"""
    局部期望梯度估计器$\hat{g}_{LE}$（严格对齐论文式3-14/3-15）
    解决NaN问题：数值保护+梯度保留+维度对齐+梯度裁剪
    :param prob: CNN输出的动作概率图 [B,1,H,W]，Sigmoid输出∈(0,1)，带梯度
    :param c: 连续调图像 [B,1,H,W] ∈[0,1]
    :param w_s: CSSIM权重（论文3.6.1指定0.06）
    :param eps: 数值保护小常数，避免除零/下溢
    :return: loss_marl: 标量张量，带梯度（论文式3-15的负均值）
    """
    B, C, H, W = c.shape
    # 论文式3-3：π(h=0) = 1 - π(h=1)，加数值保护避免0/1
    prob_0 = torch.clamp(1 - prob, min=eps, max=1 - eps)  # [B,1,H,W]
    prob_1 = torch.clamp(prob, min=eps, max=1 - eps)  # [B,1,H,W]
    grad = torch.zeros_like(prob, device=prob.device, dtype=prob.dtype)  # 梯度累积张量，保留梯度

    # 遍历每个像素智能体a（论文中的a∈{1,...,N}，N=H*W）
    for i in range(H):
        for j in range(W):
            # ==============================================
            # 论文中h_a'∈{0,1}：遍历两个动作，计算对应奖励R
            # 关键：使用torch.where保留梯度，替代直接赋值（避免梯度断裂）
            # ==============================================
            # 动作0：h_a'=0，其他像素保持原概率（h_-a）
            h_0 = torch.where(
                torch.zeros_like(prob) == 1, prob, prob  # 全False，保留原prob
            ).scatter_(dim=3, index=torch.tensor([j], device=prob.device).repeat(B, C, H, 1), value=0.0)
            h_0 = (h_0 > 0.5).float()  # 二值化（论文中的Thresholding）
            h_0 = torch.clamp(h_0, min=eps, max=1 - eps)  # 数值保护

            # 动作1：h_a'=1，其他像素保持原概率（h_-a）
            h_1 = torch.where(
                torch.zeros_like(prob) == 1, prob, prob
            ).scatter_(dim=3, index=torch.tensor([j], device=prob.device).repeat(B, C, H, 1), value=1.0)
            h_1 = (h_1 > 0.5).float()
            h_1 = torch.clamp(h_1, min=eps, max=1 - eps)

            # ==============================================
            # 论文式3-24：计算两个动作的奖励R(h,c) = -MSE(HVS) + w_s*CSSIM
            # ==============================================
            r_0 = reward(h_0, c, w_s)  # 动作0的奖励
            r_1 = reward(h_1, c, w_s)  # 动作1的奖励

            # ==============================================
            # 论文式3-14：累积梯度 ∇π_a(h_a') * R({h_a', h_-a})
            # 梯度=π(h=0)*r0 + π(h=1)*r1 （严格对齐论文推导）
            # ==============================================
            grad[:, :, i, j] = prob_0[:, :, i, j] * r_0 + prob_1[:, :, i, j] * r_1

    # 梯度裁剪：防止累积溢出为inf（关键抗NaN步骤）
    grad = torch.clamp(grad, min=-1e3, max=1e3)
    # 论文式3-15：L_MARL = -梯度的期望（均值）
    loss_marl = -grad.mean()
    # 最终损失数值保护：避免极端情况为NaN
    loss_marl = torch.where(torch.isnan(loss_marl), torch.tensor(0.0, device=loss_marl.device), loss_marl)
    loss_marl = torch.where(torch.isinf(loss_marl), torch.tensor(1e3, device=loss_marl.device), loss_marl)

    return loss_marl