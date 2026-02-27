import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image
import os
from torchvision import transforms


# -------------------------- 1. 数据集构建（核心：生成半色调标签）--------------------------
class HalftoneDataset(Dataset):
    def __init__(self, contone_dir, img_size=256, dither_matrix_size=64):
        """
        构建连续调图像→半色调图像的数据集
        :param contone_dir: 连续调图像文件夹路径（灰度图）
        :param img_size: 输入图像尺寸
        :param dither_matrix_size: 抖动矩阵尺寸（用于生成半色调标签）
        """
        self.contone_paths = [os.path.join(contone_dir, f) for f in os.listdir(contone_dir) if
                              f.endswith(('png', 'jpg'))]
        self.img_size = img_size
        self.dither_matrix = self.generate_void_and_cluster_matrix(dither_matrix_size)  # 论文提到的VAC抖动矩阵

    def generate_void_and_cluster_matrix(self, size):
        """生成论文中提到的Void-And-Cluster抖动矩阵（简化实现）"""
        matrix = np.zeros((size, size))
        energy = np.random.rand(size, size)  # 简化能量计算
        for i in range(size * size):
            # 选择能量最小（Void）或最大（Cluster）的点
            y, x = np.unravel_index(np.argmin(energy), (size, size))
            matrix[y, x] = i / (size * size)  # 归一化到[0,1]
            # 更新周围能量（简化）
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if 0 <= y + dy < size and 0 <= x + dx < size:
                        energy[y + dy, x + dx] += 0.1
        return matrix

    def contone_to_halftone(self, contone_img):
        """用抖动矩阵生成半色调图像（论文中有序抖动法的延伸，用于构建标签）"""
        h, w = contone_img.shape
        dither_h, dither_w = self.dither_matrix.shape
        halftone = np.zeros_like(contone_img)
        # 循环应用抖动矩阵
        for y in range(h):
            for x in range(w):
                dither_val = self.dither_matrix[y % dither_h, x % dither_w]
                halftone[y, x] = 1 if contone_img[y, x] > dither_val else 0
        return halftone

    def __getitem__(self, idx):
        # 读取连续调图像（灰度图）
        contone_path = self.contone_paths[idx]
        contone = Image.open(contone_path).convert('L')
        # 预处理： resize + 归一化到[0,1]
        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])
        contone_tensor = transform(contone).float()  # [1, H, W]

        # 生成半色调标签（模拟论文中DBS生成的高质量标签）
        contone_np = contone_tensor.squeeze().numpy()
        halftone_np = self.contone_to_halftone(contone_np)
        halftone_tensor = torch.from_numpy(halftone_np).unsqueeze(0).float()  # [1, H, W]

        return contone_tensor, halftone_tensor

    def __len__(self):
        return len(self.contone_paths)


# -------------------------- 2. 模型定义（cGAN：生成器+判别器）--------------------------
class UNetGenerator(nn.Module):
    """生成器：UNet架构（论文提到全卷积网络适配任意尺寸）"""

    def __init__(self, in_channels=1, out_channels=1, num_filters=64):
        super().__init__()
        # 下采样
        self.down1 = self.conv_block(in_channels, num_filters)
        self.down2 = self.conv_block(num_filters, num_filters * 2)
        self.down3 = self.conv_block(num_filters * 2, num_filters * 4)
        # 上采样
        self.up2 = self.up_conv(num_filters * 4, num_filters * 2)
        self.up1 = self.up_conv(num_filters * 2, num_filters)
        # 输出层（Sigmoid输出概率）
        self.out = nn.Conv2d(num_filters, out_channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def up_conv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 下采样
        d1 = self.down1(x)  # [B, 64, H/2, W/2]
        d2 = self.down2(d1)  # [B, 128, H/4, W/4]
        d3 = self.down3(d2)  # [B, 256, H/8, W/8]
        # 上采样
        u2 = self.up2(d3)  # [B, 128, H/4, W/4]
        u1 = self.up1(u2)  # [B, 64, H/2, W/2]
        # 输出概率图
        out = self.sigmoid(self.out(u1))  # [B, 1, H, W]
        return out


class MultiScaleDiscriminator(nn.Module):
    """判别器：多尺度马尔科夫判别器（修复：对多尺度特征做自适应池化后拼接）"""

    def __init__(self, in_channels=2):
        super().__init__()
        # 输入：连续调图像 + 半色调图像（拼接为2通道）
        self.scale1 = self.disc_block(in_channels, 64, 4, 2)  # 大感受野
        self.scale2 = self.disc_block(in_channels, 64, 2, 1)  # 小感受野
        # 池化后每个尺度为 [B,64,1,1] -> 展平后 64
        self.fc = nn.Sequential(
            nn.Linear(64 * 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def disc_block(self, in_ch, out_ch, kernel_size, stride):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, contone, halftone):
        x = torch.cat([contone, halftone], dim=1)  # [B, 2, H, W]
        s1 = self.scale1(x)  # [B, 64, H1, W1]
        s2 = self.scale2(x)  # [B, 64, H2, W2]
        # 自适应池化到 1x1，保证尺度一致
        s1_p = nn.functional.adaptive_avg_pool2d(s1, (1, 1)).view(s1.size(0), -1)  # [B, 64]
        s2_p = nn.functional.adaptive_avg_pool2d(s2, (1, 1)).view(s2.size(0), -1)  # [B, 64]
        out = self.fc(torch.cat([s1_p, s2_p], dim=1))  # [B, 1]
        return out

# -------------------------- 3. HVS滤波模块（论文重点强调的感知损失）--------------------------
class HVSFilter(nn.Module):
    """模拟人类视觉系统（HVS）低通滤波（论文中Näsänen HVS模型简化）"""

    def __init__(self, kernel_size=11, sigma=2):
        super().__init__()
        # 生成高斯低通滤波器（模拟HVS低通特性）
        kernel = self.gaussian_kernel(kernel_size, sigma)
        self.kernel = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0).float()  # [1,1,K,K]

    def gaussian_kernel(self, size, sigma):
        """生成高斯核"""
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        center = size // 2
        kernel = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel

    def forward(self, x):
        # 应用HVS滤波
        return nn.functional.conv2d(x, self.kernel, padding=self.kernel.size(-1) // 2)


# -------------------------- 4. 训练流程 --------------------------
def train_halftone_cgan(contone_dir, epochs=50, batch_size=8, lr=2e-4):
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 构建数据集
    dataset = HalftoneDataset(contone_dir=contone_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. 初始化模型、损失函数、优化器
    generator = UNetGenerator().to(device)
    discriminator = MultiScaleDiscriminator().to(device)
    hvs_filter = HVSFilter().to(device)

    # 损失函数（论文提到的L1 + 对抗损失 + HVS-MSE）
    l1_loss = nn.L1Loss()
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()

    # 优化器
    opt_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # 3. 训练循环
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        total_loss_g = 0.0
        total_loss_d = 0.0

        for contone, halftone_real in dataloader:
            contone = contone.to(device)
            halftone_real = halftone_real.to(device)
            batch_size = contone.size(0)

            # -------------------------- 训练判别器 --------------------------
            opt_d.zero_grad()
            # 生成假半色调图像（训练时用松弛阈值，避免梯度消失）
            halftone_fake = generator(contone)  # 输出概率图，未二值化
            # 判别真实样本
            pred_real = discriminator(contone, halftone_real)
            loss_d_real = bce_loss(pred_real, torch.ones_like(pred_real).to(device))
            # 判别假样本
            pred_fake = discriminator(contone, halftone_fake.detach())
            loss_d_fake = bce_loss(pred_fake, torch.zeros_like(pred_fake).to(device))
            # 总判别器损失
            loss_d = (loss_d_real + loss_d_fake) / 2
            loss_d.backward()
            opt_d.step()
            total_loss_d += loss_d.item()

            # -------------------------- 训练生成器 --------------------------
            opt_g.zero_grad()
            # 生成假半色调图像
            halftone_fake = generator(contone)
            # 对抗损失
            pred_fake = discriminator(contone, halftone_fake)
            loss_g_adv = bce_loss(pred_fake, torch.ones_like(pred_fake).to(device))
            # L1损失（像素级一致性）
            loss_g_l1 = l1_loss(halftone_fake, halftone_real)
            # HVS-MSE损失（论文重点强调的感知一致性）
            contone_hvs = hvs_filter(contone)
            halftone_fake_hvs = hvs_filter(halftone_fake)
            loss_g_hvs = mse_loss(halftone_fake_hvs, contone_hvs)
            # 总生成器损失（权重参考论文设置）
            loss_g = loss_g_adv * 0.1 + loss_g_l1 * 1.0 + loss_g_hvs * 0.5
            loss_g.backward()
            opt_g.step()
            total_loss_g += loss_g.item()

        # 打印日志
        avg_loss_d = total_loss_d / len(dataloader)
        avg_loss_g = total_loss_g / len(dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss_D: {avg_loss_d:.4f}, Loss_G: {avg_loss_g:.4f}")

    # 保存模型
    torch.save(generator.state_dict(), "halftone_generator.pth")
    print("训练完成，模型已保存为 halftone_generator.pth")


# -------------------------- 5. 推理函数（生成最终二值半色调图像）--------------------------
def infer_halftone(contone_path, model_path, img_size=256):
    """加载训练好的模型，生成二值半色调图像"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载模型
    generator = UNetGenerator().to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()

    # 预处理输入图像
    contone = Image.open(contone_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    contone_tensor = transform(contone).unsqueeze(0).float().to(device)

    # 推理：生成概率图 → 硬阈值二值化（论文中推理阶段的处理）
    with torch.no_grad():
        halftone_prob = generator(contone_tensor)
        halftone_binary = (halftone_prob > 0.5).float()  # 二值化

    # 保存结果
    halftone_np = halftone_binary.squeeze().cpu().numpy() * 255
    halftone_img = Image.fromarray(halftone_np.astype(np.uint8))
    halftone_img.save("output_halftone.png")
    print("半色调图像已保存为 output_halftone.png")


def check_contone_dir(contone_dir):
    print("当前工作目录:", os.getcwd())
    abs_path = os.path.abspath(contone_dir)
    print("解析后的路径:", abs_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(
            f"数据目录不存在: {abs_path}\n"
            f"解决方法：\n"
            f"1) 确认目录存在并填写正确路径，例如使用绝对路径:\n"
            f"   train_halftone_cgan(contone_dir=r'D:\\GraduationProject\\Code\\Halftoning\\dataset\\VOCdevkit\\VOC2012\\JPEGImages')\n"
            f"2) 或者从项目根目录启动脚本，使相对路径 `../dataset/...` 有效。\n"
            f"3) 若尚未下载数据，请先准备数据集到该目录。"
        )

# -------------------------- 6. 运行入口 --------------------------
if __name__ == "__main__":

    # 1. 训练（需提前准备连续调图像文件夹，如VOC2012的灰度图子集）
    check_contone_dir("../../dataset/VOCdevkit/VOC2012/JPEGImages")  # 检查数据目录是否存在
    train_halftone_cgan(contone_dir="../../dataset/VOCdevkit/VOC2012/JPEGImages", epochs=50)

    # 2. 推理（使用训练好的模型生成半色调图像）
    infer_halftone(
        contone_path="../../dataset/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg.png",  # 测试连续调图像路径
        model_path="halftone_generator.pth"  # 训练好的模型路径
    )