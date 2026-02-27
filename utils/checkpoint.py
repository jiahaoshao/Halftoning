import os
import torch
from util.logger import logger


def save_checkpoint(epoch, global_step, model, optimizer, scheduler, scaler, loss, filename):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'loss': loss,
    }
    torch.save(checkpoint, filename)
    logger.info(f"Checkpoint saved to {filename}")

def load_checkpoint(filename, model, optimizer, scheduler, scaler, device):
    """加载检查点，返回 epoch, global_step, loss"""
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if scaler is not None:
            scaler_state = checkpoint.get('scaler_state_dict', None)
            if scaler_state:
                try:
                    scaler.load_state_dict(scaler_state)
                except Exception as e:
                    logger.warning("Failed to load scaler state: %s. Skipping scaler.load_state_dict()", e)
            else:
                logger.info("No scaler state in checkpoint (saved disabled or None); skipping scaler load.")

        epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        loss = checkpoint['loss']
        logger.info(f"Loaded checkpoint from {filename}, resuming from epoch {epoch+1}")
        return epoch, global_step, loss
    else:
        return 0, 0, None