import json

import imageio
import numpy as np
import cv2
import os, argparse
from glob import glob

import torch

from agent.model import HalftoningPolicyNet
from utils import util
from collections import OrderedDict

from utils.dataset import get_dataloader


class Inferencer:
    def __init__(self, checkpoint_path, model):
        print(f"Loading checkpoint from {checkpoint_path}...")

        # 加载权重
        checkpoint = torch.load(checkpoint_path)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # 移除 'module.' 前缀（如果是多卡训练保存的权重）
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

        self.model = model
        self.model.load_state_dict(new_state_dict)
        self.model.eval()

    def infer(self, dataloader, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print(f"Start inference on {len(dataloader.dataset)} images...")

        with torch.no_grad():
            for i, (imgs, filenames) in enumerate(dataloader):
                imgs = imgs

                # 模型推理
                # 注意：训练时有 noise_std=0.3，推理时通常保持一致或设为0，这里使用默认逻辑
                # forward(self, cont_img, noise_img=None, noise_std=0.3)
                prob = self.model(imgs)

                # 二值化处理
                halftones = (prob > 0.5).float()

                # 保存结果
                for j in range(len(filenames)):
                    filename = filenames[j]
                    ht_tensor = halftones[j].cpu().squeeze().numpy()  # (H, W)

                    # 映射回 0-255 并转为 uint8
                    ht_img = (ht_tensor * 255).astype(np.uint8)

                    save_name = os.path.splitext(filename)[0] + '_halftone.png'
                    save_path = os.path.join(save_dir, save_name)
                    cv2.imwrite(save_path, ht_img)
                    print(f"[{i * dataloader.batch_size + j + 1}/{len(dataloader.dataset)}] Saved: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Halftoning')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('--model', default=None, type=str,
                        help='model weight file path')
    parser.add_argument('--data_dir', default=None, type=str,
                        help='where to load input data (RGB images)')
    parser.add_argument('--save_dir', default=None, type=str,
                        help='where to save the result')
    args = parser.parse_args()

    config = json.load(open(args.config))

    test_data_loader = get_dataloader(config, dtype='test')

    model = HalftoningPolicyNet()

    # 3. 初始化推理器并运行
    inferencer = Inferencer(args.model, model)
    inferencer.infer(test_data_loader, args.save_dir)