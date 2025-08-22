# !/usr/bin/env python
# -- coding: utf-8 --
# 小何一定行
import random
import torch.nn.functional as F
import numpy as np
import torch, math
import torch.nn as nn
from matplotlib import pyplot as plt
import yaml
from easydict import EasyDict
import sys

from noise_layers.identity import Identity
from noise_layers.color import Color_Manipulation
from noise_layers.blur import BLUR
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.jpeg import JPEGCompress
from noise_layers.jpegUDH import JpegCompression2
from noise_layers.noise import Noise
from noise_layers.resizeThenBack import Resize


##################
#  5种良性噪声组合 #
##################

class ComposeTest(nn.Module):
    def __init__(self, args, writer=None):
        super(ComposeTest, self).__init__()
        self.device = torch.device(args.device_ID)
        self.args = args
        self.writer = writer
        self.identity = Identity()
        self.blur = BLUR()
        self.color = Color_Manipulation
        self.noise = Noise
        self.resize = Resize
        # self.jpeg = JpegCompression(self.device)  # UDH里的JPEG压缩
        # self.jpeg = JpegCompression2(args, self.args.jpeg_quality)  # UDH改进型jpeg
        self.jpeg = JPEGCompress  # StegaStamp里的JPEG压缩

    def forward(self, img, glob_step):
        # batch = img.shape[0]
        # group = batch // 5
        # # print(group)
        # a = self.blur(img[0:group, :, :, :].clone())
        # b = self.color(self.args, img[group:2 * group, :, :, :].clone(), glob_step, self.writer)
        # # c = self.jpeg(self.args, img[2 * group:3 * group, :, :, :].clone(), glob_step, self.writer)
        # c = self.jpeg(img[2 * group:3 * group, :, :, :].clone())
        # d = self.noise(self.args, img[3 * group:4 * group, :, :, :].clone(), glob_step, self.writer)
        # e = self.identity(img[4 * group:batch, :, :, :].clone())
        # result = torch.cat([a, b, c, d, e], dim=0)
        # return result
        #
        # batch = img.shape[0]
        # group = batch // 2
        # # # print(group)
        # a = self.blur(img[0:group, :, :, :].clone())
        # a = self.color(self.args, a, glob_step, self.writer)
        # a = self.noise(self.args, a, glob_step, self.writer)
        #
        # b = self.jpeg(self.args, img[group:batch, :, :, :].clone(), glob_step, self.writer)
        # # b = self.jpeg(img[group:batch, :, :, :].clone())
        # result = torch.cat([a, b], dim=0)
        # return result

        out = self.resize(self.args, img, glob_step, self.writer)
        out = self.blur(out)
        out = self.color(self.args, out, glob_step, self.writer)
        out = self.noise(self.args, out, glob_step, self.writer)
        out = self.jpeg(self.args, out, glob_step, self.writer)
        # out = self.jpeg(out, glob_step, train_type='test')
        return out


if __name__ == '__main__':
    from PIL import Image, ImageOps
    from matplotlib import pyplot as plt
    import yaml
    from easydict import EasyDict

    with open('../args/endemask_options.yaml', 'r') as f:
        args = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))

    ComposeTest = ComposeTest(args).to('cuda')
    glob_step = 100000
    img = torch.randn(5, 3, 256, 256).to('cuda')
    out = ComposeTest(img, glob_step)
    print(out.shape)
