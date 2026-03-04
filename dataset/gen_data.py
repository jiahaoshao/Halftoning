import os
import random
import shutil
from os.path import basename

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
val_img_dir = 'HalftoneVOC2012/val'
test_img_dir = 'HalftoneVOC2012/test'
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
clear_files_in_path(val_img_dir)
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
