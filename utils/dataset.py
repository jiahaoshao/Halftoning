import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class HalftoneVOC2012(Dataset):
    def __init__(self, npy_dir, dtype):
        self.npy_dir = npy_dir
        self.npy_names = sorted(os.listdir(npy_dir))
        self.dtype = dtype

        if self.dtype != 'test':
            self.npy_names = [f for f in self.npy_names if f.endswith('.npy')]

    def __len__(self):
        return len(self.npy_names)

    def __getitem__(self, idx):

        if self.dtype == 'test':
            img_path = os.path.join(self.npy_dir, self.npy_names[idx])
            # 【优化】使用PIL.convert('L')，比OpenCV更稳定，速度更快
            img = Image.open(img_path).convert('L')  # 转为灰度图
            img = np.array(img, dtype=np.float32) / 255.0  # 归一化到[0,1]
            img = torch.from_numpy(img).unsqueeze(0).float()
            return img, self.npy_names[idx]

        npy_path = os.path.join(self.npy_dir, self.npy_names[idx])
        # 加载.npy文件（比加载图片快10倍+）
        img_np = np.load(npy_path, allow_pickle=True)
        # 转为tensor并增加通道维度 (1, H, W)
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).float()
        # 返回tensor + 文件名（和原接口一致）
        return img_tensor, self.npy_names[idx]

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