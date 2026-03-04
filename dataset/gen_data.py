import os
import random
import shutil
from os.path import basename

import cv2
import numpy as np
from PIL import Image

from utils.util import ensure_dir

# VOC2012数据集划分：训练集 13758 张，验证集 1683 张，测试集 1684 张
# TEST_COUNT = 1684
# VAL_COUNT = 1683
# TRAIN_COUNT = 13758
TEST_COUNT = 8
VAL_COUNT = 64
TRAIN_COUNT = 64

img_dir = 'VOCdevkit/VOC2012/JPEGImages'

train_img_dir = 'HalftoneVOC2012/train'
train_npy_dir = 'HalftoneVOC2012/train_npy'
val_img_dir = 'HalftoneVOC2012/val'
val_npy_dir = 'HalftoneVOC2012/val_npy'
test_img_dir = 'HalftoneVOC2012/test'
test_npy_dir = 'HalftoneVOC2012/test_npy'
test_result_dir = 'HalftoneVOC2012/test_result'


def clear_files_in_path(target_path):
    if not os.path.exists(target_path):
        print(f"警告：路径 {target_path} 不存在，无需清理")
        return

    for root, dirs, files in os.walk(target_path, topdown=False):  # topdown=False 先处理子目录
        for file_name in files:
            file_path = os.path.join(root, file_name)
            # 删除文件
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"删除文件失败：{file_path}，错误：{str(e)}")
    print(f"清理完成！路径：{target_path}")

clear_files_in_path(train_img_dir)
clear_files_in_path(train_npy_dir)
clear_files_in_path(val_img_dir)
clear_files_in_path(val_npy_dir)
clear_files_in_path(test_img_dir)
clear_files_in_path(test_result_dir)

ensure_dir(train_img_dir)
ensure_dir(val_img_dir)
ensure_dir(test_img_dir)

imgs = [f for f in os.listdir(img_dir) if f.lower().endswith('.jpg')]
random.shuffle(imgs)

total = len(imgs)
required = TEST_COUNT + VAL_COUNT
if total < required:
    raise ValueError(f'图片数量不足: 需要至少 {required} 张，当前 {total} 张')

# 按固定数量切分：先取测试，再验证，剩余为训练
test_imgs = imgs[:TEST_COUNT]
val_imgs = imgs[TEST_COUNT:TEST_COUNT + VAL_COUNT]
train_imgs = imgs[TEST_COUNT + VAL_COUNT: TRAIN_COUNT + TEST_COUNT + VAL_COUNT]



def copy_pair_list(img_list, dst_dir):
    for img_name in img_list:
        src_img = os.path.join(img_dir, img_name)
        dst_img = os.path.join(dst_dir, basename(src_img))
        shutil.copy2(src_img, dst_img)
    print(f"已复制 {len(img_list)} 张图片到 {dst_dir}")

copy_pair_list(train_imgs, train_img_dir)
copy_pair_list(val_imgs, val_img_dir)
copy_pair_list(test_imgs, test_img_dir)


def preprocess_img_to_npy(img_path, crop_size):
    """
    单张图片预处理并返回numpy数组（和原dataset.py逻辑完全一致）
    :param img_path: 单张图片路径
    :param crop_size: 裁剪尺寸
    :return: 预处理后的numpy数组 (H, W)，float32，范围[0,1]
    """
    # 灰度化 + 归一化
    img = Image.open(img_path).convert('L')
    img = np.array(img, dtype=np.float32) / 255.0

    # train集：随机裁剪或resize到指定尺寸
    h, w = img.shape
    if h >= crop_size and w >= crop_size:
        top = np.random.randint(0, h - crop_size + 1)
        left = np.random.randint(0, w - crop_size + 1)
        img = img[top:top + crop_size, left:left + crop_size]
    else:
        img = cv2.resize(img, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)

    img = np.ascontiguousarray(img)
    return img


def batch_process_to_npy(raw_dir, save_dir, crop_size):
    """
    批量处理图片并保存为.npy
    :param raw_dir: 原始图片根目录
    :param save_dir: .npy保存根目录
    :param crop_size: 裁剪尺寸
    :param dtype: train/test
    """
    ensure_dir(save_dir)
    img_names = sorted(os.listdir(raw_dir))  # 和原代码保持排序一致

    # 批量处理 + 进度条
    for img_name in img_names:
        # 过滤非图片文件（可选，避免误处理）
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue

        img_path = os.path.join(raw_dir, img_name)
        # 预处理图片
        img_np = preprocess_img_to_npy(img_path, crop_size)
        # 生成.npy文件名（保留原名称，替换后缀）
        npy_name = os.path.splitext(img_name)[0] + ".npy"
        npy_path = os.path.join(save_dir, npy_name)
        # 保存为.npy二进制文件
        np.save(npy_path, img_np, allow_pickle=False)

    print(f"已处理 {len(img_names)} 张图片，保存到 {save_dir}")


batch_process_to_npy(
    raw_dir=train_img_dir,
    save_dir=train_npy_dir,
    crop_size=64
)

batch_process_to_npy(
    raw_dir=val_img_dir,
    save_dir=val_npy_dir,
    crop_size=64
)