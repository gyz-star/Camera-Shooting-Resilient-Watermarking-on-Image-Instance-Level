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

    def __init__(self, resize_ratio_range, interpolation_method='nearest'):
        super(RESIZE, self).__init__()
        self.interpolation_method = interpolation_method
        self.resize_ratio_min = resize_ratio_range[0]
        self.resize_ratio_max = resize_ratio_range[1]
        self.interpolation_method = interpolation_method

    def forward(self, img, mask):
        resize_ratio = random_float(self.resize_ratio_min, self.resize_ratio_max)
        assert img.shape == mask.shape
        ori_size = img.shape[2:4]
        scale_size = (int(ori_size[0] * resize_ratio), int(ori_size[1] * resize_ratio))
        result_img = F.interpolate(
            img,
            size=scale_size,
            mode=self.interpolation_method,
        )
        result_mask = F.interpolate(
            mask,
            size=scale_size,
        )
        result_mask[result_mask > 0] = 1.0
        return result_img, result_mask


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
