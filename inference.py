import json
import numpy as np
import cv2
import os
import argparse
import torch
import time
from agent.model import HalftoningPolicyNet
from collections import OrderedDict


class Inferencer:
    def __init__(self, checkpoint_path, model):
        print(f"Loading checkpoint from {checkpoint_path}...")

        # 自动选择设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # 加载权重
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # 移除前缀
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                name = k[len('_orig_mod.'):]
            elif k.startswith('module.'):
                name = k[7:]
            else:
                name = k
            new_state_dict[name] = v

        self.model = model.to(self.device)
        self.model.load_state_dict(new_state_dict)
        self.model.eval()

    def infer(self, img_tensor):
        """
        核心推理函数：接收图片 Tensor，返回生成的半色调图片 Tensor
        :param img_tensor: 输入图片 Tensor，形状应为 (1, 1, H, W)，值域 [0, 1]
        :return: 输出图片 Tensor，形状为 (1, 1, H, W)，值域 {0, 1}
        """
        # 将数据移到对应设备
        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            # 模型推理
            prob = self.model(img_tensor)
            # 二值化处理
            halftone = (prob > 0.5).float()

        # 返回结果（如果希望在外部使用，通常也可以移回 cpu，但这里先保持原样，由调用者决定）
        return halftone

    def infer_and_save(self, input_path, output_path):
        """
        便捷函数：传入图片地址和保存地址（文件夹或文件），自动完成加载、推理、保存
        :param input_path: 输入图片路径
        :param output_path: 输出文件夹路径 或 完整的文件路径
        """
        # 1. 加载并预处理图片
        img_tensor = self._load_image(input_path)

        # 2. 调用核心推理函数  
        st = time.time()
        halftone_tensor = self.infer(img_tensor)
        et = time.time()
        print(f"Inference completed in {(et - st) * 1000:.3f} ms.")

        # 3. 确定最终保存路径
        # 判断 output_path 是否为文件夹，或者是否没有后缀
        # 如果是文件夹，或者路径不存在且不含后缀，则认为是要存到该文件夹下
        is_dir = os.path.isdir(output_path)
        has_no_ext = os.path.splitext(output_path)[1] == ''

        final_save_path = output_path
        if is_dir or has_no_ext:
            # 确保文件夹存在
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            # 从 input_path 获取原始文件名
            original_name = os.path.basename(input_path)
            name, ext = os.path.splitext(original_name)

            # 生成新文件名 (例如: input.jpg -> input_halftone.png)
            # 强制使用 png 防止 jpg 压缩损失半色调细节
            new_filename = f"{name}_halftone.png"

            # 拼接路径
            final_save_path = os.path.join(output_path, new_filename)

        # 4. 后处理并保存
        self._save_image(halftone_tensor, final_save_path)

    def _load_image(self, img_path):
        """内部方法：读取图片并转为 Tensor"""
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")

        # 归一化
        img = img.astype(np.float32) / 255.0
        # 转为 Tensor: (H, W) -> (1, 1, H, W)
        img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        return img_tensor

    def _save_image(self, tensor, save_path):
        """内部方法：将 Tensor 保存为图片"""
        # 1. 确保目录存在
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 2. 检查并修复文件后缀名
        # 获取后缀 (如 '.jpg')
        ext = os.path.splitext(save_path)[1].lower()

        # 支持的格式列表
        valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

        # 如果没有后缀，或者后缀不支持，强制加上 .png
        if ext not in valid_exts:
            print(f"Warning: File extension '{ext}' not recognized or missing. Appending '.png'.")
            save_path = save_path + '.png'  # 或者你喜欢的 '.jpg'

        # 3. 后处理
        # squeeze 去掉 batch 和 channel -> (H, W)
        ht_np = tensor.cpu().squeeze().numpy()
        # 映射回 0-255
        ht_img = (ht_np * 255).astype(np.uint8)

        # 4. 保存
        success = cv2.imwrite(save_path, ht_img)
        if success:
            print(f"Image saved successfully: {save_path}")
        else:
            # 如果依然失败，尝试一个绝对安全的路径
            backup_path = os.path.join(os.getcwd(), "backup_output.png")
            cv2.imwrite(backup_path, ht_img)
            print(f"Failed to save to original path. Saved to backup instead: {backup_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Halftoning Inference')
    parser.add_argument('--model', required=True, type=str, help='model weight file path')
    parser.add_argument('--input', required=True, type=str, help='input image path')
    parser.add_argument('--output', required=True, type=str, help='output image path')
    args = parser.parse_args()

    # 初始化
    model = HalftoningPolicyNet()
    inferencer = Inferencer(args.model, model)


    # 调用便捷函数进行推理并保存
    inferencer.infer_and_save(args.input, args.output)