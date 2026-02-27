import os

import imageio
import torch
import torch.nn.functional as F
import numpy as np


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_list(save_path, data_list, append_mode=False):
    n = len(data_list)
    if append_mode:
        with open(save_path, 'a') as f:
            f.writelines([str(data_list[i]) + '\n' for i in range(n-1,n)])
    else:
        with open(save_path, 'w') as f:
            f.writelines([str(data_list[i]) + '\n' for i in range(n)])
    return None

def tensor2array(tensors):
    arrays = tensors.detach().to("cpu").numpy()
    return np.transpose(arrays, (0, 2, 3, 1))

def img2tensor(img):
    if len(img.shape) == 2:
        img = img[..., np.newaxis]
    img_t = np.expand_dims(img.transpose(2, 0, 1), axis=0)
    img_t = torch.from_numpy(img_t.astype(np.float32))
    return img_t


def tensor2img(img_t):
    img = img_t[0].detach().to("cpu").numpy()
    img = np.transpose(img, (1, 2, 0))
    if img.shape[-1] == 1:
        img = img[..., 0]
    return img

def save_images_from_batch(img_batch, save_dir, filename_list, batch_no=-1):
    N,H,W,C = img_batch.shape
    if C == 3:
        #! rgb color image
        for i in range(N):
            # [-1,1] >>> [0,255]
            img_batch_i = np.clip(img_batch[i,:,:,:]*0.5+0.5, 0, 1)
            image = (255.0*img_batch_i).astype(np.uint8)
            save_name = filename_list[i] if batch_no==-1 else '%05d.png' % (batch_no*N+i)
            imageio.imwrite(os.path.join(save_dir, save_name), image)
    elif C == 1:
        #! single-channel gray image
        for i in range(N):
            # [-1,1] >>> [0,255]
            img_batch_i = np.clip(img_batch[i,:,:,0]*0.5+0.5, 0, 1)
            image = (255.0*img_batch_i).astype(np.uint8)
            save_name = filename_list[i] if batch_no==-1 else '%05d.png' % (batch_no*img_batch.shape[0]+i)
            imageio.imwrite(os.path.join(save_dir, save_name), image)
    return None

def gaussian_kernel(size=11, sigma=1.5):
    """生成2D高斯核"""
    ax = torch.arange(size).float() - size//2
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.float()


def hvs_filter(img, kernel):
    """
    img: (B,1,H,W)
    kernel: 可以是 (K,K) 或 (1,1,K,K)
    """
    if kernel.dim() == 2:
        kernel = kernel.unsqueeze(0).unsqueeze(0).float()  # 变为 (1,1,K,K)
    elif kernel.dim() == 4:
        kernel = kernel.float()
    else:
        raise ValueError(f"Unexpected kernel dimension: {kernel.dim()}")

    pad = kernel.shape[-1] // 2
    img_pad = F.pad(img, (pad, pad, pad, pad), mode='reflect')
    filtered = F.conv2d(img_pad, kernel, bias=None, stride=1, padding=0)
    return filtered

def ssim(img1, img2, window_size=11, sigma=1.5, size_average=True):
    """计算SSIM（简化版，返回逐像素图或均值）"""
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)

    # 实现参考 https://github.com/Po-Hsun-Su/pytorch-ssim
    # 此处返回逐像素SSIM图
    kernel = gaussian_kernel(window_size, sigma).to(img1.device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1,1,11,11)

    # 计算均值
    mu1 = F.conv2d(img1, kernel, padding=window_size//2)
    mu2 = F.conv2d(img2, kernel, padding=window_size//2)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # 计算方差和协方差
    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=window_size//2) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=window_size//2) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map

def contrast_map(img, window_size=11, sigma=1.5):
    """计算对比度图σ_c (公式3-22)"""
    img = torch.clamp(img, 0, 1)
    kernel = gaussian_kernel(window_size, sigma).to(img.device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    # 局部均值
    mu = F.conv2d(img, kernel, padding=window_size//2)
    # 局部方差
    sigma_sq = F.conv2d(img * img, kernel, padding=window_size//2) - mu.pow(2)
    sigma = torch.sqrt(sigma_sq + 1e-8)
    # 归一化因子k=2，使得σ_c在[0,1]？论文未明确，但通常对比度可归一化
    # 这里简单返回sigma * 2，但需确保范围
    return sigma * 2

def cssim(img1, img2, window_size=11, sigma=1.5, size_average=True):
    """
    img1, img2: (B,1,H,W)
    若 size_average=True 返回标量，否则返回每个样本的均值 (B,)
    """
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)
    ssim_map = ssim(img1, img2, window_size, sigma, size_average=False)  # (B,1,H,W)
    contrast_c = contrast_map(img1, window_size, sigma)  # (B,1,H,W)
    contrast_c = torch.clamp(contrast_c, 0, 1)
    cssim_map = contrast_c * ssim_map + (1 - contrast_c)  # (B,1,H,W)
    if size_average:
        return cssim_map.mean()
    else:
        # 对每个样本的空间维度求平均
        return cssim_map.view(img1.size(0), -1).mean(dim=1)

def anisotropic_loss(prob_map):
    """
    计算各向异性抑制损失 L_AS (公式3-19)
    prob_map: (B,1,H,W) 概率图，值在[0,1]
    """
    prob_map = prob_map.float()
    B, _, H, W = prob_map.shape
    # 计算功率谱
    # 为简化，对每个样本单独处理
    loss = 0
    for i in range(B):
        p = prob_map[i,0]  # (H,W)
        # 傅里叶变换
        fft = torch.fft.fft2(p)
        fft_shift = torch.fft.fftshift(fft)
        power = torch.abs(fft_shift) ** 2 / (H*W)  # 功率谱
        # 计算径向平均
        # 生成频率坐标
        u = torch.arange(H).float() - H//2
        v = torch.arange(W).float() - W//2
        U, V = torch.meshgrid(u, v, indexing='ij')
        radii = torch.sqrt(U**2 + V**2).round().long()  # 距离取整
        max_radius = radii.max()
        radial_sum = torch.zeros(max_radius+1).to(p.device)
        radial_count = torch.zeros(max_radius+1).to(p.device)
        for r in range(max_radius+1):
            mask = (radii == r)
            radial_sum[r] = power[mask].sum()
            radial_count[r] = mask.sum()
        radial_mean = radial_sum / (radial_count + 1e-8)
        # 计算各向异性：每个半径上各频率的差异平方
        aniso = 0
        for r in range(max_radius+1):
            mask = (radii == r)
            if mask.sum() > 0:
                diff = (power[mask] - radial_mean[r]) ** 2
                aniso += diff.sum()
        loss += aniso
    return loss / B