#!/usr/bin/env python3
# python
"""
测试不同 num_workers 下 DataLoader 速度的脚本。

用法（在项目根目录运行）：
python tests/test_num_workers.py --data-dir <your_data_dir> --batch-size 16 --workers 0,1,2,4,8 --batches 100 --runs 3

返回每个 num_workers 的平均 images/sec 和标准差。
"""
import argparse
import os
import sys
import time
import statistics

# 确保项目根目录在 sys.path，方便导入 utils.dataset
sys.path.insert(0, os.getcwd())

import torch
from torch.utils.data import DataLoader
from utils.dataset import HalftoneVOC2012

def make_loader(data_dir, batch_size, num_workers, dtype='train', prefetch_factor=8):
    dataset = HalftoneVOC2012(data_dir, dtype)
    # Windows 推荐使用 spawn（通常为默认），但显式设置以避免多次运行冲突
    try:
        torch.multiprocessing.set_start_method('spawn', force=False)
    except RuntimeError:
        pass
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else 2,
        drop_last=False
    )
    return loader

def measure_loader(loader, batches, device='cuda' if torch.cuda.is_available() else 'cpu', warmup=10):
    # 将数据拉取到 CPU 或 GPU 以模拟实际训练开销（可选：将 tensor.to(device)）
    it = iter(loader)
    # 预热若干 batch（不计时）
    for _ in range(min(warmup, batches)):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        # 触发 pin_memory -> to(device) 开销但不做梯度
        if device != 'cpu':
            # 非阻塞搬运（依赖 pin_memory）
            imgs = batch[0].to(device, non_blocking=True)
        else:
            imgs = batch[0]

    # 正式计时
    start = time.perf_counter()
    seen_batches = 0
    while seen_batches < batches:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        imgs = batch[0]
        if device != 'cpu':
            imgs = imgs.to(device, non_blocking=True)
        seen_batches += 1
    # 等待 GPU 完成（如果使用 CUDA）
    if device != 'cpu' and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, help='数据目录（npy 或 测试图片目录），对应 HalftoneVOC2012 的初始化路径')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', default='0,1,2,4,8', help='逗号分隔的 num_workers 值列表')
    parser.add_argument('--batches', type=int, default=100, help='每次测量遍历的 batch 数')
    parser.add_argument('--runs', type=int, default=3, help='每个 num_workers 重复测量次数，取均值和 std')
    parser.add_argument('--dtype', default='train', choices=['train','test'], help='数据集类型')
    parser.add_argument('--prefetch-factor', type=int, default=8)
    args = parser.parse_args()

    workers_list = [int(x) for x in args.workers.split(',') if x.strip() != '']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'运行设备: {device}, 数据目录: `{args.data_dir}`, batch_size: {args.batch_size}')

    results = {}
    for w in workers_list:
        speeds = []
        for run in range(args.runs):
            loader = make_loader(args.data_dir, args.batch_size, w, dtype=args.dtype, prefetch_factor=args.prefetch_factor)
            # 小心：如果数据集非常小，batches 大于 dataset 长度，脚本仍会循环读取
            elapsed = measure_loader(loader, args.batches, device=device, warmup=10)
            imgs = args.batch_size * args.batches
            ips = imgs / elapsed
            speeds.append(ips)
            print(f'num_workers={w} run={run+1}/{args.runs} -> {ips:.1f} imgs/sec (elapsed {elapsed:.3f}s)')
            # 清理 loader（尤其是 persistent_workers）
            del loader
            # 小睡一会避免瞬时系统抖动叠加
            time.sleep(1.0)
        results[w] = (statistics.mean(speeds), statistics.stdev(speeds) if len(speeds) > 1 else 0.0)

    print('\n最终结果 (images/sec):')
    for w in sorted(results.keys()):
        mean, std = results[w]
        print(f' num_workers={w:2d} -> mean={mean:8.1f}  std={std:7.1f}')

if __name__ == '__main__':
    # python test\test_num_workers.py --data-dir dataset/HalftoneVOC2012/train_npy --batch-size 64 --workers 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 --dtype train
    main()
