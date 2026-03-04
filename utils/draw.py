import torch
import matplotlib.pyplot as plt

checkpoint = torch.load('../halftoning_dev/model_best.pth.tar', map_location='cpu')
loss_history = checkpoint['loss_history']

# 绘制 Total Loss 曲线
plt.plot(loss_history['total_loss'], label='Total Loss')
plt.plot(loss_history['metric'], label='Metric (PSNR+CSSIM)')
plt.legend()
plt.show()