import os
import h5py
import numpy as np
import cv2
from PIL import Image
import argparse


def preprocess_and_save(root, save_path, crop_size=64, dtype='train'):
    """
    离线预处理图像并保存到HDF5文件
    :param root: 原始数据目录
    :param save_path: 保存的HDF5文件路径（如 ./data/train_voc2012.h5）
    :param crop_size: 裁剪尺寸（和原逻辑一致）
    :param dtype: train/test（test不裁剪，保留原尺寸）
    """
    img_names = sorted(os.listdir(root))
    num_samples = len(img_names)

    # 初始化存储容器（train固定尺寸，test动态尺寸）
    if dtype == 'train':
        # train: (N, 1, crop_size, crop_size)
        img_data = np.zeros((num_samples, 1, crop_size, crop_size), dtype=np.float32)
    else:
        # test: 先收集所有图像的shape，再动态存储（避免浪费空间）
        img_list = []

    # 逐张预处理（和原Dataset逻辑完全一致）
    for idx, img_name in enumerate(img_names):
        img_path = os.path.join(root, img_name)
        img = Image.open(img_path).convert('L')  # 灰度图
        img = np.array(img, dtype=np.float32) / 255.0  # 归一化到[0,1]

        if dtype == 'train':
            h, w = img.shape
            if h >= crop_size and w >= crop_size:
                top = np.random.randint(0, h - crop_size + 1)
                left = np.random.randint(0, w - crop_size + 1)
                img_crop = img[top:top + crop_size, left:left + crop_size]
            else:
                img_crop = cv2.resize(img, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
            img_data[idx, 0] = img_crop  # 填充到张量中
        else:
            # test保留原尺寸，仅归一化+增加通道维度
            img = np.expand_dims(img, axis=0)  # (1, H, W)
            img_list.append(img)

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{num_samples} images")

    # 写入HDF5文件
    with h5py.File(save_path, 'w') as f:
        # 存储图像名称（转为bytes格式，HDF5不支持直接存字符串列表）
        img_names_bytes = [name.encode('utf-8') for name in img_names]
        f.create_dataset('img_names', data=img_names_bytes, dtype=h5py.string_dtype())

        if dtype == 'train':
            # train直接存储固定尺寸张量
            f.create_dataset('images', data=img_data, dtype=np.float32, compression='gzip')  # 轻量压缩节省空间
        else:
            # test按样本存储（每个样本一个dataset）
            grp = f.create_group('images')
            for idx, img in enumerate(img_list):
                grp.create_dataset(str(idx), data=img, dtype=np.float32)

    print(f"Preprocessing done! Saved to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='原始数据目录')
    parser.add_argument('--save_path', type=str, required=True, help='HDF5保存路径')
    parser.add_argument('--crop_size', type=int, default=64, help='裁剪尺寸')
    parser.add_argument('--dtype', type=str, choices=['train', 'test'], default='train')
    args = parser.parse_args()

    preprocess_and_save(args.root, args.save_path, args.crop_size, args.dtype)