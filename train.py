import argparse
import datetime
import json
import time
import torch
import torch.optim as optim
import os
from torch.backends import cudnn
from agent.loss import HalftoneMARLLoss, TotalHalftoneLoss
from utils.logger import setup_logging
from utils.dataset import get_dataloader
from utils.util import ensure_dir, save_list, tensor2array, save_images_from_batch
from agent.model import HalftoningPolicyNet

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
        self.model = HalftoningPolicyNet().to(self.device)

        ## optimizer
        self.optimizer = getattr(optim, config['optimizer_type'])(self.model.parameters(), **config['optimizer'])
        self.lr_scheduler = getattr(optim.lr_scheduler, config['lr_scheduler_type'])(self.optimizer, **config['lr_scheduler'])

        ## dataset loader
        self.train_data_loader = get_dataloader(self.config, train=True)
        self.logger.info("Training samples: %d", len(self.train_data_loader.dataset))


        # 混合精度训练
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        self.total_loss_fn = TotalHalftoneLoss(ws=0.06, wa=0.002)

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
            self.logger.info("[*] --- epoch: %d/%d | loss: %4.4f | metric: %4.4f | time-consumed: %4.2fs ---", epoch + 1, self.n_epochs, epoch_loss['total_loss'], epoch_metric, time.time() - ep_st)

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

            self.train_data_loader = get_dataloader(self.config, train=True)

        self.logger.info("Training finished! consumed %f sec", time.time() - start_time)


    def _train_epoch(self, epoch):
        self.model.train()
        epoch_marl_loss = 0.0
        epoch_las_loss = 0.0
        epoch_total_loss = 0.0

        for batch_idx, (c, _) in enumerate(self.train_data_loader):
            # 混合精度上下文
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                c = c.to(self.device)
                z = torch.randn_like(c) * 0.3
                prob = self.model(c, z)  # 网络原始输出 (B,1,H,W)

                B, C, H, W = c.shape
                cg = torch.rand(B, C, H, W).uniform_(0, 1).to(self.device)  # 恒定灰度图
                zg = torch.randn(B, C, H, W).to(self.device)
                prob_cg = self.model(cg, zg)  # 恒定灰度图的输出 (B,1,H,W)

                # 计算损失
                total_loss, marl_loss, las_loss = self.total_loss_fn(prob, c, z, prob_cg)

            # 反向传播
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                self.optimizer.step()

            # 记录损失
            epoch_marl_loss += marl_loss.item()
            epoch_las_loss += las_loss.item()
            epoch_total_loss += total_loss.item()

            if batch_idx % self.save_freq == 0:
                self.logger.info("[%d/%d] iter:%d loss:%4.4f ", epoch + 1, self.n_epochs, batch_idx + 1, total_loss.item())

        epoch_loss = dict()
        epoch_loss['marl_loss'] = epoch_marl_loss / len(self.train_data_loader)
        epoch_loss['las_loss'] = epoch_las_loss / len(self.train_data_loader)
        epoch_loss['total_loss'] = epoch_total_loss / len(self.train_data_loader)

        return epoch_loss


    def _valid_epoch(self, epoch):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_idx, (c, _) in enumerate(self.train_data_loader):
                c = c.to(self.device)
                z = torch.randn_like(c) * 0.3
                prob = self.model(c, z)  # 网络原始输出 (B,1,H,W)

                B, C, H, W = c.shape
                cg = torch.rand(B, C, H, W).uniform_(0, 1).to(self.device)  # 恒定灰度图
                zg = torch.randn(B, C, H, W).to(self.device)
                prob_cg = self.model(cg, zg)  # 恒定灰度图的输出 (B,1,H,W)

                # 计算损失
                loss, marl_loss, las_loss = self.total_loss_fn(prob, c, z, prob_cg)
                total_loss += loss.item()

                h = (prob > 0.5).float()
                h_imgs = tensor2array(h)
                save_images_from_batch(h_imgs, self.val_halftone, None, batch_idx)

                self.logger.info("Validation: [%d/%d] iter:%d loss:%4.4f " % (epoch + 1, self.n_epochs, batch_idx + 1, loss.item()))

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