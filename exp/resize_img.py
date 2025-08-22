# !/usr/bin/env python
# -- coding: utf-8 --
# 小何一定行
import random
import sys

sys.path.append("..")
from torchvision import transforms
from PIL import Image, ImageOps
import torchgeometry
import glob
import os
import cv2
import numpy as np
from kornia import losses
import torch.nn.functional as F
from matplotlib import pyplot as plt
import torch


def main(dir_path=None, save_dir=None, img_size=256):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, default=dir_path)
    parser.add_argument('--save_dir', type=str, default=save_dir)
    args_this = parser.parse_args()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    files_list = glob.glob(args_this.images_dir + '/*')

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), Image.BICUBIC),
        transforms.ToTensor(), ])

    with torch.no_grad():
        for i, filename in enumerate(files_list):
            image = transform(Image.open(filename).convert('RGB')).unsqueeze(0)
            name = filename.split('\\')[-1] if filename.find('/') == -1 else filename.split('/')[-1]
            save_name = name.split('.')[0]
            slash = '\\' if filename.find('/') == -1 else '/'

            save_path_result = args_this.save_dir + slash + save_name + '.jpg'

            save_result = transforms.ToPILImage()(image.squeeze(0))
            save_result.save(save_path_result, quality=100)

        # grid_show = torch.cat(grid, dim=0)
        # grid_save = torchvision.utils.make_grid(grid_show, nrow=1, padding=5)
        # path = save_path + '/psnr29.png'
        # torchvision.utils.save_image(grid_save, fp=path, nrow=1, padding=5)

        # result_partial = partialIMG.detach().cpu().numpy()
        # result_partial = np.transpose(result_partial, (0, 2, 3, 1))
        # result_partial = (np.clip(result_partial[0], 0.0, 1.0) * 255).astype(np.uint8)
        # result_partial = cv2.cvtColor(result_partial, cv2.COLOR_RGB2BGR)
        #
        # result_partialMask = partialMask.detach().cpu().numpy()
        # result_partialMask = np.transpose(result_partialMask, (0, 2, 3, 1))
        # result_partialMask = (np.clip(result_partialMask[0], 0.0, 1.0) * 255).astype(np.uint8)
        # result_partialMask = cv2.cvtColor(result_partialMask, cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    img_size = 400
    dir_path = '/home/yizhi/exp/sample/visual quality/origion'
    save_path = '/home/yizhi/exp/sample/visual quality/origion'
    main(dir_path, save_path, img_size)
