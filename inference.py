import imageio
import numpy as np
import cv2
import os, argparse
from glob import glob

import torch

from agent.model import HalftoningPolicyNet
from utils import util
from collections import OrderedDict



class Inferencer:
    def __init__(self, checkpoint_path, model, use_cuda=False):
        self.checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.raw_state = self.checkpoint.get('state_dict', self.checkpoint)
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()

        ## remove keyword "module" in the state_dict
        state_dict = OrderedDict()
        for k, v in self.raw_state.items():
            name = k
            if name.startswith('module.'):
                name = name[7:]
            if name.startswith('.'):
                name = name[1:]
            state_dict[name] = v

        self.model.load_state_dict(state_dict)

    def __call__(self, c):
        torch.manual_seed(131)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(131)
        with torch.no_grad():
            c = c.to(self.device)
            prob = self.model(c)  # 网络原始输出 (B,1,H,W)
            h = (prob > 0.5).float()
        return h


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Halftoning')
    parser.add_argument('--model', default=None, type=str,
                        help='model weight file path')
    parser.add_argument('--data_dir', default=None, type=str,
                        help='where to load input data (RGB images)')
    parser.add_argument('--save_dir', default=None, type=str,
                        help='where to save the result')
    args = parser.parse_args()

    halftoning = Inferencer(
        checkpoint_path=args.model,
        model=HalftoningPolicyNet()
    )
    save_dir = os.path.join(args.save_dir)
    util.ensure_dir(save_dir)
    test_imgs = glob(os.path.join(args.data_dir, '*.*g'))
    print('------loaded %d images.' % len(test_imgs) )
    for img in test_imgs:
        print('[*] processing %s ...' % img)
        input_img = cv2.imread(img, flags=cv2.IMREAD_GRAYSCALE) / 127.5 - 1.
        h = halftoning(util.img2tensor(input_img))
        h_img = util.tensor2img(h)  # 期望返回 0..1，值为 0 或 1
        h_img = (h_img > 0.5).astype(np.uint8) * 255  # 二值化并映射为 uint8 的 0/255
        filename = os.path.join(save_dir, 'halftone_' + os.path.splitext(os.path.basename(img))[0] + '.png')
        cv2.imwrite(filename, h_img)
        print('[*] saved %s.' % filename)