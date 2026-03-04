import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class HalftoneVOC2012(Dataset):
    def __init__(self, root, dtype, crop_size=64):
        self.root = root
        self.img_names = sorted(os.listdir(self.root))
        self.crop_size = crop_size
        self.dtype = dtype

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.img_names[idx])
        # 【优化】使用PIL.convert('L')，比OpenCV更稳定，速度更快
        img = Image.open(img_path).convert('L')  # 转为灰度图
        img = np.array(img, dtype=np.float32) / 255.0  # 归一化到[0,1]

        if self.dtype == 'test':
            img = torch.from_numpy(img).unsqueeze(0).float()
            return img, self.img_names[idx]

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
    dataset = HalftoneVOC2012(config[f'{dtype}_data_dir'], dtype)
    # 【优化】针对Windows系统+16核CPU优化DataLoader参数
    if dtype == 'test':
        dataloader = DataLoader(dataset, batch_size=1,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=True)
    else:
        # 【优化】适配16核CPU，num_workers设为8，prefetch_factor提升至8，最大化预加载效率
        num_workers = config['data_loader'].get('num_workers', 8)
        dataloader = DataLoader(
            dataset,
            batch_size=config['data_loader']['batch_size'],
            shuffle=config['data_loader']['shuffle'],
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=config['data_loader'].get('prefetch_factor', 8),
            multiprocessing_context='spawn',
            drop_last=True  # 【优化】drop_last=True，避免最后一个batch尺寸不一致导致的编译重优化
        )
    return dataloader