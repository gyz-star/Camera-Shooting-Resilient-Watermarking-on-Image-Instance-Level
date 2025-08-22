# !/usr/bin/env python
# -- coding: utf-8 --
# 小何一定行
##################
#    高斯噪声    #
##################
import random
import torch.nn.functional as F
import numpy as np
import torch, math
import torch.nn as nn
from matplotlib import pyplot as plt
import yaml
from easydict import EasyDict


def Noise(img, rnd_noise):
    device = img.device
    # rnd_noise = torch.rand(1)[0] * rnd_noise  # rnd_noise:0.02
    noise = torch.normal(mean=0, std=rnd_noise, size=img.size(), dtype=torch.float32)
    if torch.cuda.is_available():
        noise = noise.to(device)
    img = img + noise  # 实际上也就是原图+噪音
    img = torch.clamp(img, 0, 1)
    return img


if __name__ == '__main__':
    from PIL import Image, ImageOps

    with open('../args/endemask_options.yaml', 'r') as f:
        args = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))
    img = Image.open('/backup/mingjin/pycharm/Data/coco/train2017/000000007281.jpg')
    img = np.array(img) / 255.
    img_r = np.transpose(img, [2, 0, 1])
    img_tensor = torch.from_numpy(img_r).unsqueeze(0).float()
    print(img_tensor.shape)
    out = Noise(args, img_tensor, 0)
    # out = blur(img_tensor)
    out = np.transpose(out.detach().squeeze(0).numpy(), [1, 2, 0])
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(out)
    plt.show()
