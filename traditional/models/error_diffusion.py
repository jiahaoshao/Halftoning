"""
误差扩散算法实现
"""
import numpy as np
import torch
from PIL import Image


def error_diffusion(gray_tensor, kernel_type='floyd-steinberg'):
    """
    处理PyTorch灰度图张量的误差扩散算法
    :param gray_tensor: 输入灰度图张量，形状为(1, H, W)，值范围[0,1]
    :param kernel_type: 误差扩散核类型，支持 'floyd-steinberg'（默认）、'jarvis-judice-ninke'
    :return: 二值化张量（形状(1, H, W)，值为0或1）和PIL图像
    """

    # 转换为[0,255]范围的numpy数组（深拷贝避免修改原始数据）
    gray_np = (gray_tensor.squeeze(0).cpu().numpy() * 255).astype(np.float64)
    h, w = gray_np.shape

    # 定义误差扩散核（相对坐标和权重，(dx, dy, weight)）
    if kernel_type == 'floyd-steinberg':
        # 最常用的Floyd-Steinberg核（右、下、右下、左下）
        kernel = [(1, 0, 7 / 16), (0, 1, 5 / 16), (1, 1, 3 / 16), (-1, 1, 1 / 16)]
    elif kernel_type == 'jarvis-judice-ninke':
        # 更复杂的核，效果更细腻（但计算量稍大）
        kernel = [
            (1, 0, 7 / 48), (2, 0, 5 / 48),
            (-1, 1, 3 / 48), (0, 1, 5 / 48), (1, 1, 7 / 48), (2, 1, 5 / 48),
            (-2, 2, 1 / 48), (-1, 2, 3 / 48), (0, 2, 5 / 48), (1, 2, 3 / 48), (2, 2, 1 / 48)
        ]
    else:
        raise ValueError("不支持的核类型，请选择 'floyd-steinberg' 或 'jarvis-judice-ninke'")

    # 逐像素处理（从左到右，从上到下）
    for i in range(h):
        for j in range(w):
            # 当前像素值（可能已被之前的误差修改）
            current_pixel = gray_np[i, j]

            # 二值化：大于127.5（中间值）设为白（255），否则为黑（0）
            quantized = 255.0 if current_pixel > 127.5 else 0.0

            # 计算量化误差（原始值 - 量化值）
            error = current_pixel - quantized

            # 将误差扩散到周围未处理的像素
            for dx, dy, weight in kernel:
                ni, nj = i + dy, j + dx  # 注意：dy是行方向偏移，dx是列方向偏移
                # 确保扩散的像素在图像范围内
                if 0 <= ni < h and 0 <= nj < w:
                    gray_np[ni, nj] += error * weight

            # 更新当前像素为量化后的值
            gray_np[i, j] = quantized

    # 转换为二值化张量（0或1，形状(1, H, W)）
    dithered_np = (gray_np / 255.0).astype(np.float64)
    dithered_tensor = torch.tensor(dithered_np, dtype=torch.float64).unsqueeze(0)

    # 转换为PIL图像（便于保存/显示）
    dithered_img = Image.fromarray(gray_np.astype(np.uint8), mode='L')

    return dithered_tensor, dithered_img