import torch.nn.functional as F


def compute_local_expected_gradient(net, c, z, hvs_filter, w_s=0.06):
    """
    计算局部期望梯度（\hat{g}_{LE}）
    net: 策略模型
    c: 连续调图像 (B,1,H,W)
    z: 高斯噪声 (B,1,H,W)
    hvs_filter: HVS低通滤波核 (K,K)
    w_s: CSSIM权重
    """
    B, _, H, W = c.shape
    device = c.device
    grad = None

    # 1. 采样初始半色调图像 h ~ Bernoulli(prob)
    prob = net(c, z)  # (B,1,H,W)
    h = torch.bernoulli(prob).detach()  # (B,1,H,W), 离散动作

    # 2. 对每个像素a，计算翻转动作后的奖励
    for a_h in range(H):
        for a_w in range(W):
            # 翻转像素(a_h,a_w)的动作
            h_flip = h.clone()
            h_flip[:, :, a_h, a_w] = 1 - h_flip[:, :, a_h, a_w]

            # 计算奖励 R(h,c) = -MSE(HVS(h), HVS(c)) + w_s*CSSIM(h,c)
            # HVS滤波（用卷积实现）
            h_hvs = F.conv2d(h_flip, hvs_filter.unsqueeze(0).unsqueeze(0), padding=hvs_filter.shape[0] // 2)
            c_hvs = F.conv2d(c, hvs_filter.unsqueeze(0).unsqueeze(0), padding=hvs_filter.shape[0] // 2)
            mse = F.mse_loss(h_hvs, c_hvs, reduction='mean')

            # 计算CSSIM（简化实现，完整版需按公式计算局部对比度）
            cssim = 1.0  # 此处替换为实际CSSIM计算代码
            reward = -mse + w_s * cssim

            # 计算梯度：∇logπ(h_a'|c,z) * R
            log_prob_flip = torch.log(prob * h_flip + (1 - prob) * (1 - h_flip))  # (B,1,H,W)
            loss = -reward * log_prob_flip.mean()  # 负奖励（梯度上升转下降）

            # 累加梯度
            if grad is None:
                grad = torch.autograd.grad(loss, net.parameters())
            else:
                grad = tuple(g1 + g2 for g1, g2 in zip(grad, torch.autograd.grad(loss, net.parameters())))

    # 3. 平均梯度（按Batch和像素数）
    grad = tuple(g / (B * H * W) for g in grad)
    return grad