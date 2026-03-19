# 测试代码片段
import torch
import torch.nn.functional as F
from agent.loss import le_gradient_estimator, hvs_filter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
c = torch.rand(1,1,64,64).to(device)  # 连续调
h = torch.randint(0,2,(1,1,64,64)).float().to(device)  # 随机半色调
prob = torch.ones_like(h) * 0.5  # 模拟概率

# 用您的 le_gradient_estimator 计算 delta_Rmse
loss, _ = le_gradient_estimator(c, prob)  # 注意 w_s=0

# 手动计算某像素翻转前后的 MSE
h_hvs = hvs_filter(h)
c_hvs = hvs_filter(c)
mse_orig = F.mse_loss(h_hvs, c_hvs).item()

# 选择一个像素位置 (x,y)
x, y = 32, 32
h_new = h.clone()
h_new[0,0,x,y] = 1 - h[0,0,x,y]  # 翻转
h_new_hvs = hvs_filter(h_new)
mse_new = F.mse_loss(h_new_hvs, c_hvs).item()
delta_mse = mse_new - mse_orig
print(f"Manual ΔMSE: {delta_mse}")

# 对比您的 delta_Rmse（注意您的 delta_Rmse 是 -ΔMSE）
delta_R = -delta_mse
print(f"Expected delta_R: {delta_R}")