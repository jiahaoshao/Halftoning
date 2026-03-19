import argparse
import json
import time
import torch
import torch.optim as optim
import os
from torch.backends import cudnn
from agent.loss import le_gradient_estimator, anisotropy_suppression_loss, cssim, calculate_hvs_psnr
from utils.logger import setup_logging
from utils.dataset import get_dataloader
from utils.util import ensure_dir, save_list, tensor2array, save_images_from_batch
import torch.nn.functional as F
from agent.model import HalftoningPolicyNet

# ====================== CUDA底层配置，适配RTX系列GPU ======================
torch._dynamo.config.disable = False
os.environ["TORCH_COMPILE_DISABLE"] = "0"
# 新增：扩大编译缓存，避免缓存溢出触发重编译
torch._dynamo.config.cache_size_limit = 128
# 新增：禁用不必要的调试检查，降低CPU开销
torch._dynamo.config.suppress_errors = True
torch.autograd.set_detect_anomaly(False)

# TF32精度配置，兼顾速度与精度，适配RTX系列GPU
torch.backends.cuda.matmul.fp32_precision = "tf32"
torch.backends.cudnn.fp32_precision = "tf32"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# cudnn性能优化
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

class Trainer:
    def __init__(self, config, resume):
        self.config = config
        self.name = config['name']
        self.resume_path = resume
        self.n_epochs = config['trainer']['epochs']
        self.with_cuda = config['cuda'] and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.with_cuda else "cpu")
        self.use_amp = config['trainer']['use_amp']
        self.seed = config['seed']
        self.start_epoch = 0
        self.save_freq = config['trainer']['save_epochs']
        self.val_freq = config['trainer'].get('val_epochs', 1)
        self.log_freq = config['trainer']['log_epochs']
        self.checkpoint_dir = os.path.join(config['save_dir'], self.name)
        ensure_dir(self.checkpoint_dir)
        json.dump(config, open(os.path.join(self.checkpoint_dir, 'config.json'), 'w'),
                  indent=4, sort_keys=False)
        self.logger = setup_logging(os.path.join(self.checkpoint_dir, 'train_log'))
        self.logger.info("@Workspace: %s *************", self.checkpoint_dir)
        self.cache = os.path.join(self.checkpoint_dir, 'train_cache')
        self.val_halftone = os.path.join(self.cache, 'halftone')
        ensure_dir(self.val_halftone)

        # 论文超参数 固定
        self.w_a = 0.002  # 论文式3-20 LAS损失权重
        self.w_s = 0.06    # 论文式3-24 CSSIM权重

        # 策略网络初始化
        self.model = HalftoningPolicyNet().to(self.device)
        self.compile_mode = config['trainer'].get('compile_mode', 'reduce-overhead')
        if config['trainer'].get('enable_compile', True):
            self.model = torch.compile(self.model, mode=self.compile_mode)
            self.logger.info(f"@Model: torch.compile enabled with mode={self.compile_mode} *************")

        # 优化器与学习率调度器 严格对齐论文
        self.optimizer = optim.Adam(self.model.parameters(), **config['optimizer'])
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **config['lr_scheduler'])

        # 数据集加载
        self.train_data_loader = get_dataloader(self.config, dtype='train')
        self.val_data_loader = get_dataloader(self.config, dtype='val')
        self.logger.info("Training samples: %d", len(self.train_data_loader.dataset))
        self.logger.info("Validation samples: %d", len(self.val_data_loader.dataset))

        # 混合精度训练
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        self.monitor_best = -float('inf')
        self.loss_history = {}
        self.global_step = 0

        # 断点续训
        if self.resume_path:
            self.load_checkpoint(self.resume_path)

        # 设备信息打印
        if self.with_cuda:
            current_device_idx = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device_idx)
            gpu_memory = torch.cuda.get_device_properties(current_device_idx).total_memory / 1024 ** 3
            self.logger.info(f"@Device: Using GPU [{current_device_idx}] - {gpu_name} (Total Memory: {gpu_memory:.2f}GB) *************")
        else:
            self.logger.info("@Device: Using CPU *************")

    def _train(self):
        """训练主循环 100%对齐论文算法3.1"""
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        start_time = time.time()

        for epoch in range(self.start_epoch, self.n_epochs):
            ep_st = time.time()
            epoch_loss = self._train_epoch(epoch)
            self.lr_scheduler.step()
            epoch_lr = self.optimizer.state_dict()['param_groups'][0]['lr']

            # 验证流程
            avg_psnr, avg_cssim = 0.0, 0.0
            if (epoch + 1) % self.val_freq == 0 or epoch == (self.n_epochs - 1):
                avg_psnr, avg_cssim = self._valid_epoch(epoch)

            # 论文对齐的综合指标
            epoch_metric = avg_psnr + 40 * avg_cssim

            #  epoch日志打印
            self.logger.info("[*] --- epoch: %d/%d | loss: %.17g | metric: %.17g | lr: %.17g | time-consumed: %.17gs ---",
                             epoch + 1, self.n_epochs,
                             epoch_loss['total_loss'],
                             epoch_metric,
                             epoch_lr,
                             time.time() - ep_st)

            # 损失与指标保存
            epoch_loss['psnr'] = avg_psnr
            epoch_loss['cssim'] = avg_cssim
            epoch_loss['metric'] = epoch_metric
            epoch_loss['lr'] = epoch_lr
            self.save_loss(epoch_loss, epoch)
            for key, val in epoch_loss.items():
                if key not in self.loss_history:
                    self.loss_history[key] = []
                self.loss_history[key].append(val)

            # 模型保存
            if (epoch+1) % self.save_freq == 0 or epoch == (self.n_epochs-1):
                self.logger.info("---------- saving model checkpoint ...")
                self.save_checkpoint(epoch)
            if self.monitor_best < epoch_metric:
                self.logger.info("---------- saving new best model ...")
                self.monitor_best = epoch_metric
                self.save_checkpoint(epoch, save_best=True)

            # if self.global_step >= 200000:
            #     print(f"已完成论文要求的{200000}次迭代，训练正常终止")
            #     break

        self.logger.info("Training finished! Total time: %.2f sec", time.time() - start_time)

    def _train_epoch(self, epoch):
        """单轮训练 严格对齐论文算法3.1"""
        self.model.train()
        epoch_marl_loss = torch.tensor(0.0, device=self.device)
        epoch_las_loss = torch.tensor(0.0, device=self.device)
        epoch_total_loss = torch.tensor(0.0, device=self.device)
        total_batch = len(self.train_data_loader)

        for batch_idx, (c, _) in enumerate(self.train_data_loader):
            c = c.to(self.device, non_blocking=True)
            B, C, H, W = c.shape

            # 网络前向传播
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                prob = self.model(c)
            # 强制FP32计算损失，避免精度问题
            prob = prob.float().contiguous()
            c = c.float().contiguous()
            # 计算LE梯度估计器损失
            marl_loss, grad_norm = le_gradient_estimator(c, prob, self.w_s)

            # ====================== 2. 优化LAS损失 论文蓝噪声特性 ======================
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                # 生成恒定灰度图cg + 固定噪声zg，修复LAS不收敛问题
                cg_gray_val = torch.rand(B, C, 1, 1, device=self.device) * 0.9 + 0.05
                cg = cg_gray_val.expand(B, C, H, W)
                prob_cg = self.model(cg)
            # 强制FP32计算LAS损失
            prob_cg = prob_cg.float().contiguous()
            las_loss = anisotropy_suppression_loss(prob_cg)

            # ====================== 3. 总损失 论文式3-20 ======================
            total_loss = marl_loss + self.w_a * las_loss

            # ====================== 4. 反向传播与参数更新 ======================
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                # 梯度裁剪，阈值调整为1.0，避免有效梯度被裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step()

            self.global_step += 1

            # ====================== 5. 日志与统计 ======================
            marl_loss = torch.nan_to_num(marl_loss.detach(), nan=0.0, posinf=10.0, neginf=-10.0)
            las_loss = torch.nan_to_num(las_loss.detach(), nan=0.0, posinf=100.0, neginf=0.0)
            total_loss = torch.nan_to_num(total_loss.detach(), nan=0.0, posinf=10.0, neginf=-10.0)

            epoch_marl_loss += marl_loss
            epoch_las_loss += las_loss
            epoch_total_loss += total_loss

            # 批量日志
            if batch_idx % self.log_freq == 0:
                self.logger.info(
                    "[Epoch %d/%d] [Batch %d/%d] | total_loss: %.17f | marl_loss: %.17f | las_loss: %.17f | grad_norm: %.17f | prob_mean: %.17f | prob_std: %.17f",
                    epoch + 1, self.n_epochs,
                    batch_idx + 1, total_batch,
                    total_loss.item(), marl_loss.item(), las_loss.item(),
                    grad_norm.item(),
                    prob.mean().item(), prob.std().item()
                )

        # 轮次平均损失
        epoch_loss = dict()
        epoch_loss['marl_loss'] = (epoch_marl_loss / total_batch).item()
        epoch_loss['las_loss'] = (epoch_las_loss / total_batch).item()
        epoch_loss['total_loss'] = (epoch_total_loss / total_batch).item()
        return epoch_loss

    def _valid_epoch(self, epoch):
        self.model.eval()
        total_psnr = 0.0
        total_cssim = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, (c, filename_list) in enumerate(self.val_data_loader):
                c = c.to(self.device, non_blocking=True)
                B, C, H, W = c.shape
                total_samples += B

                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    prob = self.model(c)
                h = (prob > 0.5).float().contiguous()

                psnr = calculate_hvs_psnr(c, h)
                cssim_score = cssim(c, h)
                total_psnr += psnr.sum().item()
                total_cssim += cssim_score.sum().item()

        avg_psnr = float(total_psnr / total_samples)
        avg_cssim = float(total_cssim / total_samples)
        return avg_psnr, avg_cssim

    def save_loss(self, epoch_loss, epoch):
        """损失历史保存"""
        if epoch == 0:
            for key in epoch_loss:
                save_list(os.path.join(self.cache, key), [epoch_loss[key]], append_mode=False)
        else:
            for key in epoch_loss:
                save_list(os.path.join(self.cache, key), [epoch_loss[key]], append_mode=True)

    def load_checkpoint(self, checkpt_path):
        """断点续训"""
        self.logger.info("-loading checkpoint from: {} ...".format(checkpt_path))
        checkpoint = torch.load(checkpt_path, map_location=self.device, weights_only=False)
        self.start_epoch = checkpoint['epoch'] + 1
        self.monitor_best = checkpoint['monitor_best']
        self.global_step = checkpoint['global_step']

        # 权重加载
        state_dict = checkpoint['state_dict']
        model_is_compiled = hasattr(self.model, "_orig_mod")
        weight_has_prefix = list(state_dict.keys())[0].startswith('_orig_mod.')
        if model_is_compiled:
            target_model = self.model._orig_mod
            if weight_has_prefix:
                state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            self.logger.info(f"-model is compiled, load weights to original module")
        else:
            if weight_has_prefix:
                state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            target_model = self.model
            self.logger.info(f"-model is not compiled, load weights directly")

        load_result = target_model.load_state_dict(state_dict, strict=False)
        if load_result.missing_keys:
            self.logger.warning(f"-missing keys: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            self.logger.warning(f"-unexpected keys: {load_result.unexpected_keys}")
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'lr_scheduler' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            self.logger.info(f"-lr_scheduler state loaded, current step: {self.lr_scheduler.last_epoch}")
        if 'scaler' in checkpoint and self.use_amp:
            self.scaler.load_state_dict(checkpoint['scaler'])
        if 'loss_history' in checkpoint:
            self.loss_history = checkpoint['loss_history']
            self.logger.info(f"-loaded loss history with {len(list(self.loss_history.values())[0])} epochs.")
        else:
            self.logger.info("-no loss history found in checkpoint.")
        self.logger.info("-pretrained checkpoint loaded.")

    def save_checkpoint(self, epoch, save_best=False):
        """模型保存"""
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'monitor_best': self.monitor_best,
            'scaler': self.scaler.state_dict() if self.use_amp else None,
            'loss_history': self.loss_history,
            'global_step': self.global_step
        }
        save_path = os.path.join(self.checkpoint_dir, 'model_last.pth.tar')
        if save_best:
            save_path = os.path.join(self.checkpoint_dir, 'model_best.pth.tar')
        torch.save(state, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Halftoning')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    args = parser.parse_args()

    config_dict = json.load(open(args.config))
    trainer = Trainer(config_dict, resume=args.resume)
    trainer._train()