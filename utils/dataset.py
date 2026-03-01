import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class HalftoneVOC2012(Dataset):
    def __init__(self, root, crop_size=64):
        self.root = root
        self.img_names = sorted(os.listdir(self.root))
        self.crop_size = crop_size

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.img_names[idx])
        img = Image.open(img_path).convert('L')  # 转为灰度图
        img = np.array(img, dtype=np.float32) / 255.0  # 归一化到[0,1]

        h, w = img.shape
        if h >= self.crop_size and w >= self.crop_size:
            top = np.random.randint(0, h - self.crop_size + 1)
            left = np.random.randint(0, w - self.crop_size + 1)
            img = img[top:top+self.crop_size, left:left+self.crop_size]
        else:
            # 若图像小于裁剪尺寸，先resize
            img = cv2.resize(img, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)

        # 转换为tensor，并增加通道维度 (1, H, W)
        img = torch.from_numpy(img).unsqueeze(0).float()
        return img, self.img_names[idx]

def get_dataloader(config, dtype='train'):
    dataset = HalftoneVOC2012(config[f'{dtype}_data_dir'])
    dataloader = DataLoader(dataset, batch_size=config['data_loader']['batch_size'],
                            shuffle=config['data_loader']['shuffle'], num_workers=config['data_loader']['num_workers'], pin_memory=True)
    return dataloader