import os
import csv
import json

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm


# ===================== 1. 配置参数（全局统一管理）=====================
class Config:
    # 数据配置
    data_root = "../data"  # MNIST数据集下载路径
    save_dir = "../out/saved_mnist_files"  # 训练数据保存路径
    img_size = 28  # MNIST图像尺寸（28x28）
    batch_size = 64  # 批次大小（RTX 2080Ti可设64，CPU设16）
    num_workers = 4  # 数据加载线程数（CPU核心数//2）

    # 模型配置
    in_channels = 1  # 输入通道数（灰度图=1）
    num_classes = 10  # 输出类别数（0-9共10类）

    # 训练配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 优先GPU
    epochs = 10  # 训练轮次（MNIST简单，10轮足够收敛）
    initial_lr = 1e-3  # 初始学习率
    lr_scheduler_T = epochs  # 余弦退火周期（与epochs一致）

    # 保存配置
    save_model_name = "mnist_lenet5_params.pth"  # 模型参数文件名
    save_log_name = "training_log.csv"  # 训练日志文件名
    save_pred_name = "val_pred_results.json"  # 验证集预测结果文件名


# 初始化配置
config = Config()
os.makedirs(config.save_dir, exist_ok=True)
os.makedirs(config.data_root, exist_ok=True)

# ===================== 2. 数据加载（直接用torchvision原生MNIST）=====================
# 数据预处理：转为张量+归一化（MNIST全局均值0.1307，标准差0.3081）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载训练集（原生MNIST，无路径）
train_dataset = datasets.MNIST(
    root=config.data_root, train=True, transform=transform, download=True
)
train_loader = DataLoader(
    train_dataset, batch_size=config.batch_size, shuffle=True,
    num_workers=config.num_workers, pin_memory=True
)

# 加载验证集（原生MNIST，无路径，用索引替代路径标识样本）
val_dataset = datasets.MNIST(
    root=config.data_root, train=False, transform=transform, download=True
)
val_loader = DataLoader(
    val_dataset, batch_size=config.batch_size, shuffle=False,
    num_workers=config.num_workers, pin_memory=True
)


# ===================== 3. 模型定义（LeNet5简化版，不变）=====================
class LeNet5(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(LeNet5, self).__init__()
        # 卷积块1：28x28→28x28→14x14
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 卷积块2：14x14→10x10→5x5
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层：16x5x5→400→120→84→10
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        return x


# ===================== 4. 训练辅助函数（适配无路径场景）=====================
def save_training_log(logs, save_path):
    """保存训练日志到CSV文件（轮次、训练损失、验证准确率）"""
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train_Avg_Loss", "Val_Accuracy"])
        for log in logs:
            writer.writerow([log["epoch"], round(log["train_loss"], 4), round(log["val_acc"], 4)])
    print(f"训练日志已保存至：{save_path}")


def save_val_pred_results(model, val_loader, val_dataset, save_path, device):
    """保存验证集预测结果（用索引+图像数据替代路径，支持后续错误分析）"""
    model.eval()
    pred_results = []
    sample_idx = 0  # 用全局索引标识每个验证集样本（替代原路径）

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Saving validation predictions"):
            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)

            # 模型预测
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)

            # 转换为CPU数据（图像转为列表格式，便于JSON保存）
            imgs_np = imgs.cpu().numpy()  # (batch_size, 1, 28, 28)
            labels_np = labels.cpu().numpy()
            preds_np = preds.cpu().numpy()

            # 逐样本记录：索引、真实标签、预测标签、图像数据（简化版，可用于后续可视化）
            for img_np, true_label, pred_label in zip(imgs_np, labels_np, preds_np):
                # 图像数据展平为列表（JSON不支持numpy数组，展平后占空间小）
                img_flat = img_np.flatten().tolist()
                pred_results.append({
                    "sample_idx": sample_idx,  # 样本索引（后续可通过val_dataset[sample_idx]获取原样本）
                    "true_label": int(true_label),
                    "pred_label": int(pred_label),
                    "is_correct": int(true_label == pred_label),
                    "img_flat": img_flat  # 展平的图像数据（后续可恢复为28x28）
                })
                sample_idx += 1

    # 保存到JSON文件
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(pred_results, f, indent=2)
    print(f"验证集预测结果已保存至：{save_path}")
    return pred_results


def plot_and_save_results(train_logs, val_loader, model, device, save_dir):
    """绘制训练损失曲线+验证集数字预测结果（无路径依赖）"""
    # 1. 准备预测数据（取验证集第一批）
    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        imgs, labels = batch
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)

        # 转换为CPU数据
        imgs_np = imgs.cpu().numpy()
        labels_np = labels.cpu().numpy()
        preds_np = preds.cpu().numpy()

    # 2. 绘制图像
    plt.figure(figsize=(10, 6))

    # 子图1：训练损失曲线
    plt.subplot(1, 2, 1)
    epochs = [log["epoch"] for log in train_logs]
    train_losses = [log["train_loss"] for log in train_logs]
    plt.plot(epochs, train_losses, marker="o", color="darkblue", linewidth=2, markersize=6)
    plt.xlabel("Training Epoch", fontsize=12)
    plt.ylabel("Average Training Loss", fontsize=12)
    plt.title("MNIST Training Loss Curve", fontsize=13, pad=15)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # 子图2：验证集数字预测（前5张）
    # plt.subplot(1, 2, 2)
    # plt.subplots_adjust(wspace=0.5)
    # plt.show()

    plt.figure(figsize=(16, 6))

    for i in range(5):
        ax = plt.subplot(1, 5, i + 1)
        ax.imshow(imgs_np[i][0], cmap="gray", aspect="equal")
        ax.set_title(f"True: {labels_np[i]}\nPred: {preds_np[i]}", fontsize=11, y=-0.2)
        ax.axis("off")

    plt.tight_layout()
    save_path = os.path.join(save_dir, "mnist_results.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"可视化结果已保存至：{save_path}")


# ===================== 5. 核心训练流程（不变）=====================
def train_mnist():
    # 1. 初始化模型、损失函数、优化器
    model = LeNet5(in_channels=config.in_channels, num_classes=config.num_classes).to(config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.initial_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.lr_scheduler_T)

    # 2. 记录训练日志
    train_logs = []

    # 3. 开始训练
    print(f"开始训练（设备：{config.device}）")
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_train_loss = 0.0

        # 遍历训练集
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}"):
            imgs, labels = batch
            imgs, labels = imgs.to(config.device), labels.to(config.device)

            # 前向传播
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            # 反向传播+更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累计损失
            total_train_loss += loss.item() * imgs.size(0)

        # 本轮平均训练损失
        avg_train_loss = total_train_loss / len(train_loader.dataset)

        # 4. 验证
        model.eval()
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                imgs, labels = batch
                imgs, labels = imgs.to(config.device), labels.to(config.device)

                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)

                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

        # 验证准确率
        val_acc = total_correct / total_samples
        train_logs.append({"epoch": epoch, "train_loss": avg_train_loss, "val_acc": val_acc})

        # 打印日志
        print(
            f"Epoch {epoch:2d} | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # 学习率调度
        scheduler.step()

    # 5. 保存所有数据
    print("\n训练结束，开始保存数据...")
    # 5.1 保存模型参数
    model_save_path = os.path.join(config.save_dir, config.save_model_name)
    torch.save(model.state_dict(), model_save_path)
    print(f"模型参数已保存至：{model_save_path}")

    # 5.2 保存训练日志
    log_save_path = os.path.join(config.save_dir, config.save_log_name)
    save_training_log(train_logs, log_save_path)

    # 5.3 保存验证集预测结果（传入val_dataset用于后续索引映射）
    pred_save_path = os.path.join(config.save_dir, config.save_pred_name)
    save_val_pred_results(model, val_loader, val_dataset, pred_save_path, config.device)

    # 5.4 可视化结果
    plot_and_save_results(train_logs, val_loader, model, config.device, config.save_dir)

    return model, train_logs

def infer_new_image(model, img, device=config.device):
    """
    仅适配 val_dataset[0] 返回的 img（已被 transform 的 torch.Tensor）。
    确保输入为 (1, C, H, W)、float、在 device 上，然后推理返回 int 预测。
    """
    model.to(device)
    model.eval()

    if not isinstance(img, torch.Tensor):
        raise TypeError("expect img to be a torch.Tensor from val_dataset[0]")

    tensor = img

    # 常见情况：tensor.shape == (1, H, W) -> (1, C, H, W)
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    elif tensor.dim() == 3:
        # 如果是 (C, H, W) -> 加 batch 维
        if tensor.size(0) in (1, 3):
            tensor = tensor.unsqueeze(0)  # (1,C,H,W)
        else:
            # 如果是 (H, W, C)（不太可能），把通道移前并加 batch
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)

    # 确保浮点并移动到 device
    if not tensor.is_floating_point():
        tensor = tensor.float().div(255.0)
    tensor = tensor.to(device)

    with torch.no_grad():
        output = model(tensor)
        _, pred = torch.max(output, dim=1)
    return int(pred.item())


# ===================== 6. 执行训练 =====================
if __name__ == "__main__":
    # trained_model, training_logs = train_mnist()
    # print("\n所有流程完成！可在 ./saved_mnist_files 目录查看保存的数据。")

    # 加载模型
    model = LeNet5()
    model.load_state_dict(torch.load("../out/saved_mnist_files/mnist_lenet5_params.pth"))
    model.eval()

    print(model)

    img, target = val_dataset[0]

    print(infer_new_image(model, img))


