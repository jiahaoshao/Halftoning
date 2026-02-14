import torch.nn as nn

class simplecnn(nn.Module):
    def __init__(self, num_class): # num_class是分类数
        super().__init__()
        self.features = nn.Sequential( # 做特征提取
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2), # 保持图像大小不变 16 * 224 * 224
            nn.ReLU(), # 卷积之后接上激活函数 增加非线性特征
            nn.MaxPool2d(kernel_size=2, stride=2), # 池化层 将图像大小减半 16 * 112 * 112
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=1), # 保持图像大小不变 32 * 112 * 112
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 池化层 将图像大小减半 32 * 56 * 56
        )

        #定义全链接层 做分类
        self.classifier = nn.Sequential(
            nn.Linear(32*56*56, 128),
            nn.ReLU(),
            nn.Linear(128, num_class) # num_class分类数
        )

    def forward(self, x):
        # 前向传播部分
        x = self.features(x) # 图像特征提取
        x = x.view(x.size(0), -1) # 展平操作 x.size(0) 为 batch
        x = self.classifier(x) # 分类部分
        return x




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