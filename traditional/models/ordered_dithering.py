"""
有序抖动算法实现
"""
import numpy as np
import torch
from PIL import Image


def ordered_dithering(gray_tensor, dither_matrix=None):
    """
    处理PyTorch灰度图张量的有序抖动算法
    :param gray_tensor: 输入灰度图张量，形状为(1, H, W)，值范围[0,1]（来自VOC数据集）
    :param dither_matrix: 抖动矩阵，默认使用8x8 Bayer矩阵
    :return: 二值化张量（形状(1, H, W)，值为0或1）和PIL图像（便于保存/显示）
    """
    # 1. 验证输入格式（确保是单通道灰度图张量）
    if gray_tensor.dim() != 3 or gray_tensor.shape[0] != 1:
        raise ValueError("输入必须是形状为(1, H, W)的单通道灰度图张量")

    # 2. 将张量转换为[0,255]范围的numpy数组（便于计算）
    # 移除通道维度 (1, H, W) → (H, W)，并从[0,1]缩放至[0,255]
    gray_np = (gray_tensor.squeeze(0).cpu().numpy() * 255).astype(np.float64)
    h, w = gray_np.shape

    # 3. 定义抖动矩阵（默认8x8 Bayer矩阵）
    if dither_matrix is None:
        dither_matrix = np.array([
            [0, 32, 8, 40, 2, 34, 10, 42],
            [48, 16, 56, 24, 50, 18, 58, 26],
            [12, 44, 4, 36, 14, 46, 6, 38],
            [60, 28, 52, 20, 62, 30, 54, 22],
            [3, 35, 11, 43, 1, 33, 9, 41],
            [51, 19, 59, 27, 49, 17, 57, 25],
            [15, 47, 7, 39, 13, 45, 5, 37],
            [63, 31, 55, 23, 61, 29, 53, 21]
        ], dtype=np.float64)
        # 归一化矩阵至[0,255]（Bayer矩阵原始范围是0-63）
        dither_matrix = dither_matrix * (255.0 / 63.0)

    m, n = dither_matrix.shape  # 抖动矩阵尺寸（如8x8）

    # 4. 应用有序抖动算法
    dithered_np = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            # 循环取抖动矩阵中的阈值（矩阵平铺到图像大小）
            threshold = dither_matrix[i % m, j % n]
            # 像素值 > 阈值 → 白色（1.0），否则 → 黑色（0.0）
            dithered_np[i, j] = 1.0 if gray_np[i, j] > threshold else 0.0

    # 5. 转换回PyTorch张量格式（恢复通道维度 (H, W) → (1, H, W)）
    dithered_tensor = torch.tensor(dithered_np, dtype=torch.float64).unsqueeze(0)

    # 6. 转换为PIL图像（便于保存或显示，值范围[0,255]）
    dithered_img = Image.fromarray((dithered_np * 255).astype(np.uint8), mode='L')

    return dithered_tensor, dithered_img


