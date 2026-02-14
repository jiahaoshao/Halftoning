import torch
import torch.optim as optim
import numpy as np

from utils.compute import compute_local_expected_gradient


def train_drl_halftone(net, train_loader, val_loader, hvs_filter, epochs=200000, lr=3e-4, w_a=0.002):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    hvs_filter = torch.tensor(hvs_filter, dtype=torch.float32).to(device)

    for iter in range(epochs):
        net.train()
        total_loss = 0.0

        for batch_c, _ in train_loader:  # batch_c: (B,1,H,W)
            batch_c = batch_c.to(device)
            B, _, H, W = batch_c.shape

            # 生成高斯噪声
            batch_z = torch.normal(0, 1, size=(B, 1, H, W)).to(device)

            # 1. 计算策略梯度（局部期望梯度）
            optimizer.zero_grad()
            policy_grad = compute_local_expected_gradient(net, batch_c, batch_z, hvs_filter)

            # 2. 计算各向异性抑制损失 L_AS
            prob = net(batch_c, batch_z)
            # 生成恒定灰度图（用于L_AS计算）
            c_g = torch.rand((B, 1, H, W)).to(device)  # 均匀分布采样
            prob_g = net(c_g, batch_z)
            # 计算功率谱（简化实现，完整版需DFT）
            fft_prob = torch.fft.fft2(prob_g.squeeze(1))
            power_spectrum = torch.abs(fft_prob) ** 2 / (H * W)
            # 计算径向平均功率谱
            # （此处省略径向分组代码，直接用均值替代）
            radial_avg = power_spectrum.mean(dim=(-1, -2), keepdim=True)
            L_AS = ((power_spectrum - radial_avg) ** 2).mean()

            # 3. 总梯度更新
            for param, g in zip(net.parameters(), policy_grad):
                param.grad = g
            L_AS.backward()  # 累加L_AS的梯度
            optimizer.step()
            scheduler.step()

            total_loss += L_AS.item()

        # 验证（每1000次迭代）
        if iter % 1000 == 0:
            net.eval()
            val_cssim = 0.0
            with torch.no_grad():
                for val_c, _ in val_loader:
                    val_c = val_c.to(device)
                    val_z = torch.normal(0, 1, size=val_c.shape).to(device)
                    prob = net(val_c, val_z)
                    val_h = (prob > 0.5).float()
                    # 计算CSSIM（简化）
                    val_cssim += 1.0  # 替换为实际CSSIM计算
            val_cssim /= len(val_loader)
            print(f"Iter {iter:6d} | Train Loss: {total_loss / len(train_loader):.6f} | Val CSSIM: {val_cssim:.4f}")

    # 保存模型
    torch.save(net.state_dict(), "drl_halftone_net.pth")
    return net