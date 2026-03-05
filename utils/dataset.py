import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

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

        self.memory_cache = []
        if self.dtype != 'test':
            print(f"预加载数据集到内存: {len(self.npy_names)} 个样本")
            for f_name in self.npy_names:
                npy_path = os.path.join(self.npy_dir, f_name)
                img_np = np.load(npy_path, allow_pickle=True)
                img_tensor = torch.from_numpy(img_np).unsqueeze(0).float()
                self.memory_cache.append(img_tensor)
            print("数据集预加载完成！")

    def __len__(self):
        return len(self.npy_names)

    def __getitem__(self, idx):
        if self.dtype == 'test':
            img_path = os.path.join(self.npy_dir, self.npy_names[idx])
            img = Image.open(img_path).convert('L')
            img = np.array(img, dtype=np.float32) / 255.0
            img = torch.from_numpy(img).unsqueeze(0).float()
            return img, self.npy_names[idx]

        return self.memory_cache[idx], self.npy_names[idx]
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