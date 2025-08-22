# !/usr/bin/env python
# -- coding: utf-8 --
# 小何一定行
##################
#    色彩偏移    #
##################
import random
import torch.nn.functional as F
import numpy as np
import torch, math
import torch.nn as nn
from matplotlib import pyplot as plt
import yaml
from easydict import EasyDict


def get_rnd_brightness_torch(rnd_bri, rnd_hue, batch_size):
    # 为从[−0.1，0.1]均匀采样的每个rgb通道添加随机颜色偏移量。
    rnd_hue = torch.FloatTensor(batch_size, 3, 1, 1).uniform_(-rnd_hue, rnd_hue)
    rnd_brightness = torch.FloatTensor(batch_size, 1, 1, 1).uniform_(-rnd_bri, rnd_bri)  # [−0.3，0.3]均匀采样
    return rnd_hue + rnd_brightness


def Color_Manipulation(args, img, glob_step=100000, writer=None):
    ramp_fn = lambda ramp: np.min([glob_step / ramp, 1.])
    device = img.device
    h, w = img.shape[2:4]
    batch_size = img.shape[0]
    # 亮度和对比度: 用m∼U[0.5，1.5]和b∼U[−0.3，0.3]对mx+b进行仿射直方图重新缩放。
    # 色度偏移：为从[−0.1，0.1]均匀采样的每个rgb通道添加随机颜色偏移量。
    # 亮度和色度：
    rnd_bri = ramp_fn(args.rnd_bri_ramp) * args.rnd_bri  # rnd_bri:0.3  亮度
    rnd_hue = ramp_fn(args.rnd_hue_ramp) * args.rnd_hue  # rnd_hue:0.1  色调/色度
    rnd_brightness = get_rnd_brightness_torch(rnd_bri, rnd_hue, batch_size)
    # 对比度:
    contrast_low = 1. - (1. - args.contrast_low) * ramp_fn(args.contrast_ramp)  # contrast_low:0.5
    contrast_high = 1. + (args.contrast_high - 1.) * ramp_fn(args.contrast_ramp)  # contrast_high:1.5
    contrast_params = [contrast_low, contrast_high]  # [0.5,1.5]
    # 去饱和度：在完整的RGB图像和其灰度等效图像之间随机线性内插.此处是获得比例因子，从[0,1]均匀采样
    rnd_sat = torch.rand(1)[0] * ramp_fn(args.rnd_sat_ramp) * args.rnd_sat  # rnd_sat: (0,1.0)

    # contrast & brightness：
    # 用m∼U[0.5，1.5]和b∼U[−0.3，0.3]对mx+b进行仿射直方图重新缩放。
    # 此处是缩放因子的获取
    contrast_scale = torch.Tensor(img.size()[0]).uniform_(contrast_params[0], contrast_params[1])  # 从[0.5,1.5]均匀采样
    contrast_scale = contrast_scale.reshape(img.size()[0], 1, 1, 1)
    contrast_scale = contrast_scale.to(device)
    rnd_brightness = rnd_brightness.to(device)
    # 对mx+b进行仿射直方图重新缩放。
    img = img * contrast_scale
    img = img + rnd_brightness
    img = torch.clamp(img, 0, 1)

    # saturation
    # 去饱和度：在完整的RGB图像和其灰度等效图像之间随机线性内插。rnd_sat：（0，1）
    # 在颜色通道上进行加权求平均得到完整RGB图像对应的灰度等效图像,因求平均会降维,故在颜色通道增维
    sat_weight = torch.FloatTensor([0.3, 0.6, 0.1]).reshape(1, 3, 1, 1)
    sat_weight = sat_weight.to(device)
    image_lum = torch.sum(img * sat_weight, dim=1).unsqueeze(1)
    img = (1 - rnd_sat) * img + rnd_sat * image_lum
    img = img.reshape([-1, 3, h, w])

    if writer is not None:
        if glob_step:
            writer.add_scalar('transformer/rnd_bri', rnd_bri, glob_step)
            writer.add_scalar('transformer/rnd_sat', rnd_sat, glob_step)
            writer.add_scalar('transformer/rnd_hue', rnd_hue, glob_step)
            writer.add_scalar('transformer/contrast_low', contrast_low, glob_step)
            writer.add_scalar('transformer/contrast_high', contrast_high, glob_step)
    return img


if __name__ == '__main__':
    from PIL import Image, ImageOps

    with open('../args/ende_options.yaml', 'r') as f:
        args = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))
    img = Image.open('/backup/mingjin/pycharm/Data/coco/train2017/0000000000666.jpg')
    img = np.array(img) / 255.
    img_r = np.transpose(img, [2, 0, 1])
    img_tensor = torch.from_numpy(img_r).unsqueeze(0).float()
    print(img_tensor.shape)
    out = Color_Manipulation(args, img_tensor)
    # out = blur(img_tensor)
    out = np.transpose(out.detach().squeeze(0).numpy(), [1, 2, 0])
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(out)
    plt.show()
