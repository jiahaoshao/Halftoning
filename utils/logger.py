import logging
import os
import sys
import time

from utils.util import ensure_dir


def setup_logging(log_dir):
    """配置日志记录器"""
    ensure_dir(log_dir)
    log_filename = os.path.join(log_dir, f'train_{time.strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)