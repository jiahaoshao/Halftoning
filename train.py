import argparse
import json
import time
import torch
import torch.optim as optim
import os

from profilehooks import profile
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
torch.backends.cuda.enable_cudnn_conv_heuristic = True

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
        self.val_freq = config['trainer'].get('val_epochs', 1)  # 可配置验证频率
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

        # 策略网络初始化
        self.model = HalftoningPolicyNet().to(self.device)
        self.compile_mode = config['trainer'].get('compile_mode', 'max-autotune')
        if config['trainer'].get('enable_compile', True):
            self.model = torch.compile(self.model, mode=self.compile_mode)
            self.logger.info(f"@Model: torch.compile enabled with mode={self.compile_mode} *************")

        # 优化器与学习率调度器
        self.optimizer = optim.Adam(self.model.parameters(), **config['optimizer'])
        self.lr_scheduler = getattr(optim.lr_scheduler, config['lr_scheduler_type'])(self.optimizer, **config['lr_scheduler'])

        # 数据集加载
        self.train_data_loader = get_dataloader(self.config, dtype='train')
        self.val_data_loader = get_dataloader(self.config, dtype='val')
        self.logger.info("Training samples: %d", len(self.train_data_loader.dataset))

        # 混合精度训练
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        self.monitor_best = -float('inf')
        self.loss_history = {}

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

    @profile
    def _train(self):
        """训练主循环，完全保留论文训练流程"""
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
            epoch_metric = avg_psnr + avg_cssim

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
                self.logger.info("---------- saving model ...")
                self.save_checkpoint(epoch)
            if self.monitor_best < epoch_metric:
                self.logger.info("---------- saving best model ...")
                self.monitor_best = epoch_metric
                self.save_checkpoint(epoch, save_best=True)

        self.logger.info("Training finished! consumed %.17g sec", time.time() - start_time)

    def _train_epoch(self, epoch):
        """单轮训练，消除强制同步阻塞"""
        self.model.train()
        # ✅ 用GPU张量累加loss，避免每个batch同步
        epoch_marl_loss = torch.tensor(0.0, device=self.device)
        epoch_las_loss = torch.tensor(0.0, device=self.device)
        epoch_total_loss = torch.tensor(0.0, device=self.device)
        total_batch = len(self.train_data_loader)

        for batch_idx, (c, _) in enumerate(self.train_data_loader):
            c = c.to(self.device, non_blocking=True)  # non_blocking配合pin_memory异步拷贝
            B, C, H, W = c.shape

            # 网络前向传播，混合精度加速
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                prob = self.model(c)
                # ✅ 修复batch_size，避免重编译
                cg_gray_val = torch.rand(1, C, 1, 1, device=self.device) * 0.9 + 0.05
                cg = cg_gray_val.expand(B, C, H, W)
                prob_cg = self.model(cg)

            # ✅ 仅在需要打印日志时，才执行.item()同步，其余时间不触发同步
            if batch_idx % self.log_freq == 0:
                self.logger.info(
                    "prob_cg stats: min=%.17g max=%.17g mean=%.17g std=%.17g",
                    prob_cg.min().item(), prob_cg.max().item(),
                    prob_cg.mean().item(), prob_cg.std().item()
                )

            # 损失计算强制用FP32，避免精度问题
            prob = prob.float().contiguous()
            prob_cg = prob_cg.float().contiguous()
            c = c.float().contiguous()

            # 论文核心损失计算
            loss_st = time.time()
            marl_loss, grad_norm = le_gradient_estimator(c, prob)
            loss_ed = time.time()
            las_loss = anisotropy_suppression_loss(prob_cg)
            total_loss = marl_loss + 0.002 * las_loss  # 论文式3-20

            # 反向传播与梯度更新
            self.optimizer.zero_grad(set_to_none=True)
            if self.use_amp:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step()

            # ✅ GPU端无同步累加loss，仅做detach，不触发CPU-GPU同步
            marl_loss = torch.nan_to_num(marl_loss.detach(), nan=0.0, posinf=10.0, neginf=-10.0)
            las_loss = torch.nan_to_num(las_loss.detach(), nan=0.0, posinf=100.0, neginf=0.0)
            total_loss = torch.nan_to_num(total_loss.detach(), nan=0.0, posinf=10.0, neginf=-10.0)

            epoch_marl_loss += marl_loss
            epoch_las_loss += las_loss
            epoch_total_loss += total_loss

            # ✅ 仅日志打印时触发一次同步，其余batch不执行
            if batch_idx % self.log_freq == 0:
                self.logger.info(
                    f"LE loss computed in {loss_ed - loss_st:.17g} sec, marl_loss={marl_loss.item():.17g}, grad_norm={grad_norm:.17g}")
                self.logger.info("[%d/%d] iter:%d loss:%.17g ",
                                 epoch + 1, self.n_epochs,
                                 batch_idx + 1,
                                 total_loss.item())

        # ✅ epoch结束后，一次性同步到CPU，仅触发1次同步
        epoch_loss = dict()
        epoch_loss['marl_loss'] = (epoch_marl_loss / total_batch).item()
        epoch_loss['las_loss'] = (epoch_las_loss / total_batch).item()
        epoch_loss['total_loss'] = (epoch_total_loss / total_batch).item()
        return epoch_loss

    def _valid_epoch(self, epoch):
        """验证流程，基于skimage官方API计算指标，完全对齐论文表3.2"""
        self.model.eval()
        total_psnr = 0.0
        total_cssim = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, (c, filename_list) in enumerate(self.val_data_loader):
                c = c.to(self.device, non_blocking=True)
                B, C, H, W = c.shape
                total_samples += B

                # 模型推理
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    prob = self.model(c)
                h = (prob > 0.5).float()  # 论文指定0.5阈值二值化

                # 【skimage优化】论文对齐的HVS-PSNR计算
                psnr = calculate_hvs_psnr(c, h)
                # CSSIM计算
                cssim_score = cssim(c, h)

                # 指标累加
                total_psnr += psnr * B
                total_cssim += cssim_score.sum().item()

        # 指标平均
        avg_psnr = float(total_psnr / total_samples)
        avg_cssim = float(total_cssim / total_samples)
        self.logger.info(f"Validation Epoch {epoch + 1}: Avg PSNR={avg_psnr!r}, Avg CSSIM={avg_cssim!r}")
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
        """断点续训，兼容torch.compile后的模型权重"""
        self.logger.info("-loading checkpoint from: {} ...".format(checkpt_path))
        checkpoint = torch.load(checkpt_path, map_location=self.device, weights_only=False)
        self.start_epoch = checkpoint['epoch'] + 1
        self.monitor_best = checkpoint['monitor_best']

        # 权重兼容处理
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

        # 权重加载
        load_result = target_model.load_state_dict(state_dict, strict=False)
        if load_result.missing_keys:
            self.logger.warning(f"-missing keys: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            self.logger.warning(f"-unexpected keys: {load_result.unexpected_keys}")

        # 优化器与混合精度状态加载
        self.optimizer.load_state_dict(checkpoint['optimizer'])
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
            'monitor_best': self.monitor_best,
            'scaler': self.scaler.state_dict() if self.use_amp else None,
            'loss_history': self.loss_history
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
    node = Trainer(config_dict, resume=args.resume)
    node._train()