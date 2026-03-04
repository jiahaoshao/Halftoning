import argparse
import json
import time
import torch
import torch.optim as optim
import os
from torch.backends import cudnn
from agent.loss import le_gradient_estimator, anisotropy_suppression_loss, hvs_filter, cssim, EPS, HVS_KERNEL_SIZE, \
    HVS_SCALE
from utils.logger import setup_logging
from utils.dataset import get_dataloader
from utils.util import ensure_dir, save_list, tensor2array, save_images_from_batch
import torch.nn.functional as F
from agent.model import HalftoningPolicyNet

# os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

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
        self.checkpoint_dir = os.path.join(config['save_dir'], self.name)
        ensure_dir(self.checkpoint_dir)
        json.dump(config, open(os.path.join(self.checkpoint_dir, 'config.json'), 'w'),
                  indent=4, sort_keys=False)
        self.logger = setup_logging(os.path.join(self.checkpoint_dir, 'train_log'))
        self.logger.info("@Workspace: %s *************", self.checkpoint_dir)
        self.cache = os.path.join(self.checkpoint_dir, 'train_cache')
        self.val_halftone = os.path.join(self.cache, 'halftone')
        ensure_dir(self.val_halftone)

        ## model
        self.model = HalftoningPolicyNet().to(self.device)

        ## optimizer
        self.optimizer = getattr(optim, config['optimizer_type'])(self.model.parameters(), **config['optimizer'])
        self.lr_scheduler = getattr(optim.lr_scheduler, config['lr_scheduler_type'])(self.optimizer, **config['lr_scheduler'])

        ## dataset loader
        self.train_data_loader = get_dataloader(self.config, dtype='train')
        self.val_data_loader = get_dataloader(self.config, dtype='val')
        self.logger.info("Training samples: %d", len(self.train_data_loader.dataset))


        # 混合精度训练
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp, device=self.device)

        if self.resume_path:
            self.load_checkpoint(self.resume_path)

        # ================== 新增：设备信息记录 Start ==================
        if self.with_cuda:
            current_device_idx = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device_idx)
            self.logger.info(f"@Device: Using GPU [{current_device_idx}] - {gpu_name} *************")
        else:
            self.logger.info("@Device: Using CPU *************")


    def _train(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        cudnn.benchmark = True

        start_time = time.time()
        self.monitor_best = -float('inf')

        for epoch in range(self.start_epoch, self.n_epochs + 1):
            ep_st = time.time()
            epoch_loss = self._train_epoch(epoch)
            # perform lr_scheduler
            self.lr_scheduler.step()
            epoch_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            epoch_metric = self._valid_epoch(epoch)
            # ========== 修改点1：epoch汇总日志，移除所有小数位限制，全精度打印 ==========
            self.logger.info("[*] --- epoch: %d/%d | loss: %.17g | metric: %.17g | lr: %.17g | time-consumed: %.17gs ---",
                             epoch + 1, self.n_epochs,
                             epoch_loss['total_loss'],
                             epoch_metric,
                             epoch_lr,
                             time.time() - ep_st)
            # save losses and learning rate
            epoch_loss['metric'] = epoch_metric
            epoch_loss['lr'] = epoch_lr
            self.save_loss(epoch_loss, epoch)
            if (epoch+1) % self.save_freq == 0 or epoch == (self.n_epochs-1):
                self.logger.info("---------- saving model ...")
                self.save_checkpoint(epoch)
            if self.monitor_best < epoch_metric:
                self.logger.info("---------- saving best model ...")
                self.monitor_best = epoch_metric
                self.save_checkpoint(epoch, save_best=True)

        self.logger.info("Training finished! consumed %.17g sec", time.time() - start_time)


    def _train_epoch(self, epoch):
        self.model.train()
        epoch_marl_loss = 0.0
        epoch_las_loss = 0.0
        epoch_total_loss = 0.0

        for batch_idx, (c, _) in enumerate(self.train_data_loader):
            c = c.to(self.device)
            B, C, H, W = c.shape
            # 仅网络前向传播使用autocast，损失计算用FP32保证精度
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                prob = self.model(c)
                cg_gray_val = torch.rand(B, C, 1, 1, device=self.device) * 0.9 + 0.05
                cg = cg_gray_val.expand(B, C, H, W)
                prob_cg = self.model(cg)

                if batch_idx % 100 == 0:
                    # 监控恒定灰度图输出的概率分布，std不能趋近于0
                    self.logger.info(
                        "prob_cg stats: min=%.17g max=%.17g mean=%.17g std=%.17g",
                        prob_cg.min().item(), prob_cg.max().item(),
                        prob_cg.mean().item(), prob_cg.std().item()
                    )

            # 损失计算强制用FP32，避免精度问题
            prob = prob.float()
            prob_cg = prob_cg.float()
            c = c.float()
            marl_loss, grad_norm = le_gradient_estimator(c, prob)
            las_loss = anisotropy_suppression_loss(prob_cg)
            total_loss = marl_loss + 0.002 * las_loss

            # 反向传播
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(total_loss).backward()
                # 先反缩放梯度，再做裁剪
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step()

            # 记录损失
            marl_loss = torch.nan_to_num(marl_loss, nan=0.0, posinf=10.0, neginf=-10.0)
            las_loss = torch.nan_to_num(las_loss, nan=0.0, posinf=100.0, neginf=0.0)
            total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=10.0, neginf=-10.0)

            epoch_marl_loss += marl_loss.item()
            epoch_las_loss += las_loss.item()
            epoch_total_loss += total_loss.item()

            if batch_idx % 100 == 0:
                # ========== 修改点2：训练batch日志，移除小数位限制 ==========
                self.logger.info("[%d/%d] iter:%d loss:%.17g ",
                                 epoch + 1, self.n_epochs,
                                 batch_idx + 1,
                                 total_loss.item())

        epoch_loss = dict()
        epoch_loss['marl_loss'] = epoch_marl_loss / len(self.train_data_loader)
        epoch_loss['las_loss'] = epoch_las_loss / len(self.train_data_loader)
        epoch_loss['total_loss'] = epoch_total_loss / len(self.train_data_loader)

        return epoch_loss

    def _valid_epoch(self, epoch):
        self.model.eval()
        total_psnr = 0.0
        total_cssim = 0.0
        total_samples = 0
        PAD_CROP = HVS_KERNEL_SIZE // 2

        with torch.no_grad():
            for batch_idx, (c, _) in enumerate(self.val_data_loader):
                c = c.to(self.device)  # c本身就是[0,1]值域，无需转255
                B, C, H, W = c.shape
                total_samples += B

                prob = self.model(c)
                h = (prob > 0.5).float()  # h也是[0,1]值域的二值图

                # HVS滤波不使用缩放，信号值域[0,1]，MAX_I=1，公式完全正确
                c_hvs = hvs_filter(c, apply_visual_scale=False)
                h_hvs = hvs_filter(h, apply_visual_scale=False)

                # 裁剪边缘不变
                c_hvs_valid = c_hvs[:, :, PAD_CROP:-PAD_CROP, PAD_CROP:-PAD_CROP]
                h_hvs_valid = h_hvs[:, :, PAD_CROP:-PAD_CROP, PAD_CROP:-PAD_CROP]

                # 正确的PSNR公式，MAX_I=1，和论文表3.2的数值完全对齐
                mse = F.mse_loss(h_hvs_valid, c_hvs_valid)
                psnr = 10 * torch.log10(1.0 / (mse + EPS))  # 不会再出现inf

                # CSSIM计算不变（本身基于[0,1]值域）
                cssim_score = cssim(h, c)

                # 累加统计（不变）
                total_psnr += psnr.item() * B
                total_cssim += cssim_score.sum().item()

                h_imgs = tensor2array(h)
                save_images_from_batch(h_imgs, self.val_halftone, None, batch_idx)

                self.logger.info("Validation: [%d/%d] iter:%d PSNR:%.17g CSSIM:%.17g"
                                 % (epoch + 1, self.n_epochs,
                                    batch_idx + 1,
                                    psnr.item(),
                                    cssim_score.mean().item()))

        avg_psnr = total_psnr / total_samples
        avg_cssim = total_cssim / total_samples
        # ========== 修改点4：验证epoch汇总日志，移除小数位限制 ==========
        self.logger.info(f"Validation Epoch {epoch + 1}: Avg PSNR={avg_psnr!r}, Avg CSSIM={avg_cssim!r}")

        monitor_metric = avg_psnr + avg_cssim
        return monitor_metric

    def save_loss(self, epoch_loss, epoch):
        if epoch == 0:
            for key in epoch_loss:
                save_list(os.path.join(self.cache, key), [epoch_loss[key]], append_mode=False)
        else:
            for key in epoch_loss:
                save_list(os.path.join(self.cache, key), [epoch_loss[key]], append_mode=True)

    def load_checkpoint(self, checkpt_path):
        print("-loading checkpoint from: {} ...".format(checkpt_path))
        checkpoint = torch.load(checkpt_path, map_location=self.device)
        self.start_epoch = checkpoint['epoch'] + 1
        self.monitor_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scaler' in checkpoint and self.use_amp:
            self.scaler.load_state_dict(checkpoint['scaler'])
        print("-pretrained checkpoint loaded.")

    def save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.monitor_best,
            'scaler': self.scaler.state_dict() if self.use_amp else None  # 保存scaler
        }
        save_path = os.path.join(self.checkpoint_dir, 'model_last.pth.tar')
        if save_best:
            save_path = os.path.join(self.checkpoint_dir, 'model_best.pth.tar')
        torch.save(state, save_path)

if __name__ == '__main__':
    ## start
    # D:\Paths\Python\Python313\python.exe D:\GraduationProject\Code\Halftoning\train.py -c D:\GraduationProject\Code\Halftoning\config\config.json
    parser = argparse.ArgumentParser(description='Halftoning')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    args = parser.parse_args()
    config_dict = json.load(open(args.config))
    node = Trainer(config_dict, resume=args.resume)
    node._train()