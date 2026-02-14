import torch
from models.cnn import simplecnn

x = torch.randn(32, 3, 224, 224)  # 创建一个随机输入张量，模拟一张224x224的RGB图像
model = simplecnn(num_class=4)
output = model(x)
print(output.shape)  # 输出的形状应该是 (32, 4)，对应