import argparse
import datetime
import json
import time
import torch
import torch.optim as optim
import os
from torch.backends import cudnn
from agent.losses import COMALoss
from agent.rewards import RewardCalculator
from utils.logger import setup_logging
from utils.dataset import get_dataloader
from utils.util import gaussian_kernel, anisotropic_loss, ensure_dir, save_list, tensor2array, save_images_from_batch
from agent.model import PolicyNetwork

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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
        self.model = eval(config['model'])().to(self.device)

        ## optimizer
        self.optimizer = getattr(optim, config['optimizer_type'])(self.model.parameters(), **config['optimizer'])
        self.lr_scheduler = getattr(optim.lr_scheduler, config['lr_scheduler_type'])(self.optimizer, **config['lr_scheduler'])

        ## dataset loader
        self.train_data_loader = get_dataloader(config, train=True)
        self.logger.info("Training samples: %d", len(self.train_data_loader.dataset))

        ##

        # 混合精度训练
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        # HVS核
        self.hvs_kernel = gaussian_kernel(size=11, sigma=1.5).to(self.device)  # (11,11)

        # 奖励计算器
        self.reward_calc = RewardCalculator(self.hvs_kernel, w_s=config['w_s'])

        # 损失函数
        self.marl_loss_fn = COMALoss(self.reward_calc)

        if self.resume_path:
            assert os.path.exists(self.resume_path), 'Invalid checkpoint Path: %s' % self.resume_path
            self.load_checkpoint(self.resume_path)

    def _train(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        cudnn.benchmark = True

        start_time = time.time()
        self.monitor_best = 999.0

        for epoch in range(self.start_epoch, self.n_epochs + 1):
            ep_st = time.time()
            epoch_loss = self._train_epoch(epoch)
            # perform lr_scheduler
            self.lr_scheduler.step()
            epoch_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            epoch_metric = self._valid_epoch(epoch)
            self.logger.info("[*] --- epoch: %d/%d | loss: %4.4f | metric: %4.4f | time-consumed: %4.2fs ---", epoch + 1, self.n_epochs, epoch_loss['loss_total'], epoch_metric, time.time() - ep_st)

            # save losses and learning rate
            epoch_loss['metric'] = epoch_metric
            epoch_loss['lr'] = epoch_lr
            self.save_loss(epoch_loss, epoch)
            if (epoch+1) % self.save_freq == 0 or epoch == (self.n_epochs-1):
                self.logger.info("---------- saving model ...")
                self.save_checkpoint(epoch)
            if self.monitor_best > epoch_metric:
                self.logger.info("---------- saving best model ...")
                self.monitor_best = epoch_metric
                self.save_checkpoint(epoch, save_best=True)

        self.logger.info("Training finished! consumed %f sec", time.time() - start_time)


    def _train_epoch(self, epoch):
        self.model.train()
        epoch_loss_marl = 0.0
        epoch_loss_aniso = 0.0
        epoch_loss_total = 0.0

        for batch_idx, (c, _) in enumerate(self.train_data_loader):
            c = c.to(self.device)
            # 生成高斯噪声
            z = torch.randn_like(c) * 0.3
            z = z.to(self.device)

            # 混合精度上下文
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                self.optimizer.zero_grad()
                prob = self.model(c, z)  # 网络原始输出 (B,1,H,W)

                # 现在再采样
                h_sampled = torch.bernoulli(prob)

                # 计算MARL损失
                loss_marl = self.marl_loss_fn(prob, c, h_sampled)

                # 各向异性抑制损失（对恒定灰度图）
                B, _, H, W = c.shape
                c_gray = torch.rand(B, 1, H, W, device=self.device)
                z_gray = torch.randn_like(c_gray) * 0.3
                prob_gray = self.model(c_gray, z_gray)
                loss_aniso = anisotropic_loss(prob_gray)

                # 总损失
                loss = loss_marl + self.config['w_a'] * loss_aniso

            # 反向传播
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # 记录损失
            epoch_loss_marl += loss_marl.item()
            epoch_loss_aniso += loss_aniso.item()
            epoch_loss_total += loss.item()

            if batch_idx % self.save_freq == 0:
                self.logger.info("[%d/%d] iter:%d loss:%4.4f ", epoch + 1, self.n_epochs, batch_idx + 1, loss.item())

        epoch_loss = dict()
        epoch_loss['loss_marl'] = epoch_loss_marl / len(self.train_data_loader)
        epoch_loss['loss_aniso'] = epoch_loss_aniso / len(self.train_data_loader)
        epoch_loss['loss_total'] = epoch_loss_total / len(self.train_data_loader)

        return epoch_loss


    def _valid_epoch(self, epoch):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_idx, (c, _) in enumerate(self.train_data_loader):
                c = c.to(self.device)
                # 生成高斯噪声
                z = torch.randn_like(c) * 0.3
                z = z.to(self.device)

                prob = self.model(c, z)  # 网络原始输出 (B,1,H,W)

                # 现在再采样
                h_sampled = torch.bernoulli(prob)

                # 计算MARL损失
                loss_marl = self.marl_loss_fn(prob, c, h_sampled)

                # 各向异性抑制损失（对恒定灰度图）
                B, _, H, W = c.shape
                c_gray = torch.rand(B, 1, H, W, device=self.device)
                z_gray = torch.randn_like(c_gray) * 0.3
                prob_gray = self.model(c_gray, z_gray)
                loss_aniso = anisotropic_loss(prob_gray)

                # 总损失
                loss = loss_marl + self.config['w_a'] * loss_aniso
                total_loss += loss.item()

                h = (prob > 0.5).float()
                h_imgs = tensor2array(h)
                save_images_from_batch(h_imgs, self.val_halftone, None, batch_idx)

                print("Validation: [%d/%d] iter:%d loss:%4.4f " % (epoch + 1, self.n_epochs, batch_idx + 1, loss.item()))

            return total_loss

    def save_loss(self, epoch_loss, epoch):
        if epoch == 0:
            for key in epoch_loss:
                save_list(os.path.join(self.cache, key), [epoch_loss[key]], append_mode=False)
        else:
            for key in epoch_loss:
                save_list(os.path.join(self.cache, key), [epoch_loss[key]], append_mode=True)

    def load_checkpoint(self, checkpt_path):
        print("-loading checkpoint from: {} ...".format(checkpt_path))
        checkpoint = torch.load(checkpt_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.monitor_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print("-pretrained checkpoint loaded.")

    def save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.monitor_best
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