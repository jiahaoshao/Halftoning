import matplotlib.pyplot as plt
import numpy as np

# 读取PSNR数据
def read_txt(file_path):
    with open(file_path, 'r') as f:
        data = [float(line.strip()) for line in f if line.strip()]
    return data

# 替换为实际文件路径
psnr_data = read_txt("halftoning_dev/train_cache/psnr.txt")
cssim_data = read_txt("halftoning_dev/train_cache/cssim.txt")
total_loss = read_txt("halftoning_dev/train_cache/total_loss.txt")
lr_data = read_txt("halftoning_dev/train_cache/lr.txt")

# 绘制趋势图
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0,0].plot(psnr_data, label='PSNR')
axes[0,0].axhline(y=31, color='r', linestyle='--', label='Target:31')
axes[0,0].legend(); axes[0,0].set_title('PSNR Trend')

axes[0,1].plot(cssim_data, label='CSSIM')
axes[0,1].axhline(y=0.92, color='r', linestyle='--', label='Target:0.92')
axes[0,1].legend(); axes[0,1].set_title('CSSIM Trend')

axes[1,0].plot(total_loss, label='Total Loss')
axes[1,0].legend(); axes[1,0].set_title('Total Loss Trend')

axes[1,1].plot(lr_data, label='Learning Rate')
axes[1,1].legend(); axes[1,1].set_title('Learning Rate Trend')

plt.tight_layout()
plt.show()