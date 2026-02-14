"""
直接二值搜索法（DBS）实现
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


class HVSModel:
    """人类视觉系统（HVS）模型：高斯低通滤波器"""

    def __init__(self, scale_factor=77 , kernel_size=3, device='cuda'):
        """
        :param scale_factor: 缩放参数 S = R·D（打印分辨率×观察距离）
        :param kernel_size: 滤波器核大小（奇数）
        :param device: 计算设备（'cuda' 或 'cpu'）
        """
        self.S = scale_factor
        self.device = device
        self.kernel = self._create_gaussian_kernel(kernel_size).to(device)

    def _create_gaussian_kernel(self, kernel_size):
        """生成高斯核（模拟HVS低通特性）"""
        sigma = self.S / 100.0  # 标准差与S正相关，S越大低通越强
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=self.device)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
        kernel = kernel / torch.sum(kernel)  # 归一化
        return kernel.unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度

    def filter(self, image):
        """对图像应用HVS滤波"""
        # image应为PyTorch张量，形状(1, 1, H, W)
        image = image.unsqueeze(0).unsqueeze(0).to(self.device)
        return F.conv2d(image, self.kernel, padding=self.kernel.shape[-1] // 2)


class DBSAlgorithm:
    """直接二值搜索法（DBS）实现"""

    def __init__(self, hvs_model, max_iter=10):
        """
        :param hvs_model: HVS模型实例
        :param max_iter: 最大迭代次数（全图遍历轮数）
        """
        self.hvs = hvs_model
        self.max_iter = max_iter
        self.device = hvs_model.device

    def _compute_mse(self, hvs_h, hvs_c):
        """计算HVS滤波后半色调图与原图的均方误差（MSE）"""
        return torch.mean((hvs_h - hvs_c) ** 2)

    def initialize_halftone(self, gray_tensor):
        """用阈值法初始化半色调图像（0-1二值图）"""
        # gray_tensor已在GPU上
        return (gray_tensor > 0.5).float()

    def optimize(self, gray_tensor):
        """
        执行DBS优化
        :param gray_tensor: 输入灰度图张量，形状为(1, H, W)，值范围[0,1]，已在GPU上
        :return: 优化后的半色调图（0-1二值图）
        """
        # 添加通道维度，形状(1, 1, H, W)
        gray_tensor = gray_tensor.squeeze(0).to(self.device)
        # print("gray_tensor形状:", gray_tensor.shape)
        h, w = gray_tensor.shape[0], gray_tensor.shape[1]

        # 1. 预计算原图的HVS滤波结果（固定值）
        hvs_c = self.hvs.filter(gray_tensor)

        # 2. 初始化半色调图像并计算初始HVS滤波结果
        halftone = self.initialize_halftone(gray_tensor)
        hvs_h = self.hvs.filter(halftone)
        current_mse = self._compute_mse(hvs_h, hvs_c)

        # 3. 迭代优化（全图多轮遍历）
        for iter in range(self.max_iter):
            print(f"迭代 {iter+1}/{self.max_iter}, 当前MSE: {current_mse.item():.6f}")
            improved = False
            # 遍历每个像素（随机顺序遍历可减少局部最优陷阱）
            indices = torch.randperm(h * w, device=self.device)
            for idx in indices:
                i = idx // w
                j = idx % w
                # 尝试翻转当前像素（0→1或1→0）
                halftone[i, j] = 1 - halftone[i, j]  # 翻转

                # 局部更新HVS滤波结果（此处简化为全图重新滤波）
                new_hvs_h = self.hvs.filter(halftone)
                new_mse = self._compute_mse(new_hvs_h, hvs_c)

                # print(f"迭代 {iter + 1}/{self.max_iter}, 新MSE: {new_mse.item():.6f}")

                # 若误差减小则保留翻转，否则恢复原值
                if new_mse < current_mse:
                    current_mse = new_mse
                    hvs_h = new_hvs_h
                    improved = True
                    print(f"迭代 {iter + 1}/{self.max_iter}, 新MSE: {new_mse.item():.6f}")
                else:
                    halftone[i, j] = 1 - halftone[i, j]  # 恢复

            # 若本轮无改进则提前终止
            if not improved:
                print(f"提前收敛，迭代轮数：{iter + 1}")
                break

        halftone_np = halftone.squeeze(0).squeeze(0).cpu().numpy()
        dbs_img = Image.fromarray((halftone_np * 255).astype(np.uint8), mode='L')

        return halftone, dbs_img
