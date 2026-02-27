import torch
from utils.util import hvs_filter, cssim

class RewardCalculator:
    def __init__(self, hvs_kernel, w_s=0.06):
        """
        hvs_kernel: 预先生成的高斯核 (1,1,K,K)
        w_s: CSSIM权重
        """
        self.hvs_kernel = hvs_kernel
        self.w_s = w_s

    def compute_reward(self, h, c):
        """
        h: (B,1,H,W)
        c: (B,1,H,W)
        返回: (B,) 每个样本的奖励
        """
        h_hvs = hvs_filter(h, self.hvs_kernel)
        c_hvs = hvs_filter(c, self.hvs_kernel)

        # 逐样本计算 MSE: 对每个样本的空间维度求平均
        mse_map = (h_hvs - c_hvs) ** 2  # (B,1,H,W)
        mse_per_sample = mse_map.view(h.size(0), -1).mean(dim=1)  # (B,)

        # 逐样本计算 CSSIM
        cssim_per_sample = cssim(h, c, size_average=False)  # (B,)

        reward = -mse_per_sample + self.w_s * cssim_per_sample  # (B,)
        return reward

    def compute_reward_with_flip(self, h, c, h_sampled, pixel_indices, flip_values):
        """
        高效计算翻转指定像素后的新奖励
        参数:
            h: 原始二值图像 (B,1,H,W)
            c: 连续调图像
            h_sampled: 采样得到的动作（与h相同，但为概率采样结果）
            pixel_indices: 要翻转的像素坐标列表，每个元素为(b, a)其中a是像素索引
            flip_values: 对应翻转后的值 (0或1)
        返回:
            新奖励列表，长度等于像素数
        """
        # 预先计算原始滤波结果和误差图
        h_hvs = hvs_filter(h, self.hvs_kernel)
        c_hvs = hvs_filter(c, self.hvs_kernel)
        error_map = (h_hvs - c_hvs) ** 2
        orig_reward = -error_map.mean(dim=[1,2,3])  # (B,)

        # 对每个像素快速计算
        new_rewards = []
        # 注意：为简化，这里假设batch size=1，实际需处理batch
        # 我们可循环处理每个batch
        for idx, (b, a) in enumerate(pixel_indices):
            # 获取像素位置 (假设a是展平索引)
            H, W = h.shape[2:]
            i = a // W
            j = a % W
            delta = flip_values[idx] - h[b,0,i,j]  # 变化量
            # 更新局部窗口内的h_hvs
            # 由于hvs滤波是线性的，新h_hvs = 旧h_hvs + delta * kernel
            kernel = self.hvs_kernel.squeeze()  # (K,K)
            K = kernel.shape[0]
            pad = K // 2
            # 窗口边界
            i_start = max(i - pad, 0)
            i_end = min(i + pad + 1, H)
            j_start = max(j - pad, 0)
            j_end = min(j + pad + 1, W)
            # 对应kernel中的有效区域
            k_i_start = pad - (i - i_start)
            k_i_end = k_i_start + (i_end - i_start)
            k_j_start = pad - (j - j_start)
            k_j_end = k_j_start + (j_end - j_start)
            kernel_crop = kernel[k_i_start:k_i_end, k_j_start:k_j_end]
            # 更新h_hvs在窗口内
            h_hvs_new = h_hvs[b,0].clone()
            h_hvs_new[i_start:i_end, j_start:j_end] += delta * kernel_crop
            # 计算新误差图窗口
            error_new_window = (h_hvs_new[i_start:i_end, j_start:j_end] - c_hvs[b,0,i_start:i_end, j_start:j_end]) ** 2
            error_old_window = error_map[b,0,i_start:i_end, j_start:j_end]
            # 全局均值变化
            delta_error_sum = error_new_window.sum() - error_old_window.sum()
            new_reward = orig_reward[b] - delta_error_sum / (H*W)
            new_rewards.append(new_reward)
        return torch.stack(new_rewards)  # 与pixel_indices顺序一致