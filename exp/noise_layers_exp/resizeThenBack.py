# !/usr/bin/env python
# -- coding: utf-8 --
# 小何一定行
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


##################
#      放缩      #
##################

def random_float(min, max):
    """
    Return a random number
    :param min:
    :param max:
    :return:
    """
    return np.random.rand() * (max - min) + min


class RESIZE(nn.Module):
    """
    Resize the image. The target size is original size * resize_ratio
    """

    def __init__(self, interpolation_method='bilinear'):  # nearest
        super(RESIZE, self).__init__()
        self.interpolation_method = interpolation_method

    def forward(self, img, resize_ratio):
        # interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None)
        # 用来上采样或下采样，可以给定size或者scale_factor来进行上下采样。同时支持3D、4D、5D的张量输入。
        # 插值算法可选，最近邻、线性、双线性等等。
        ori_size = img.shape[2:4]
        scale_size = (int(ori_size[0] * resize_ratio), int(ori_size[1] * resize_ratio))
        result = F.interpolate(
            img,
            size=scale_size,
            mode=self.interpolation_method,
            align_corners=True,
        )
        result = F.interpolate(
            result,
            size=ori_size,
            mode=self.interpolation_method,
            align_corners=True,
        )
        return result


def Resize(args, img, glob_step=100000, writer=None):
    device = img.device
    resize_ratio_min = args.resize_min
    resize_ratio_max = args.resize_max
    resize_ratio = random_float(resize_ratio_min, resize_ratio_max)
    resize = RESIZE().to(device)
    result = resize(img, resize_ratio)
    # print(resize_ratio)
    if writer is not None:
        if glob_step % args.vis_iter == 0 or glob_step == 1:
            writer.add_scalar('transformer/resize_ratio', resize_ratio, glob_step)
    return result


if __name__ == '__main__':
    from PIL import Image, ImageOps
    from matplotlib import pyplot as plt
    import yaml
    from easydict import EasyDict

    with open('../args/endemask_options.yaml', 'r') as f:
        args = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))
    img = Image.open('/backup/mingjin/pycharm/Data/coco/train2017/000000000009.jpg')
    img = np.array(img) / 255.
    img_r = np.transpose(img, [2, 0, 1])
    img_tensor = torch.from_numpy(img_r).unsqueeze(0).float()
    print(img_tensor.shape)
    out = Resize(args, img_tensor)
    # out = blur(img_tensor)
    out = np.transpose(out.detach().squeeze(0).numpy(), [1, 2, 0])
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(out)
    plt.show()
