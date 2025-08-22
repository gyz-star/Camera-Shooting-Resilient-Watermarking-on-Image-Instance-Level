# !/usr/bin/env python
# -- coding: utf-8 --
# 小何一定行
import random
import torch.nn.functional as F
import numpy as np
import torch, math
import torch.nn as nn
from matplotlib import pyplot as plt


def random_blur_kernel(probs, N_blur, sigrange_gauss, sigrange_line, wmin_line):
    N = N_blur  # [3,5]
    coords = torch.from_numpy(np.stack(np.meshgrid(range(N_blur), range(N_blur), indexing='ij'), axis=-1)) - (
            0.5 * (N - 1))  # （7,7,2)    [0,6]->[-3,3]
    manhat = torch.sum(torch.abs(coords), dim=-1)  # (7, 7)

    # nothing, default
    vals_nothing = (manhat < 0.5).float()  # (7, 7)

    # gauss
    # 标准差在1到3个之间随机采样 sigrange_gauss=[1., 3.]
    sig_gauss = torch.rand(1)[0] * (sigrange_gauss[1] - sigrange_gauss[0]) + sigrange_gauss[0]
    vals_gauss = torch.exp(-torch.sum(coords ** 2, dim=-1) / 2. / sig_gauss ** 2)

    # line
    theta = torch.rand(1)[0] * 2. * np.pi  # 对一个随机角度进行采样，2Π即360度
    v = torch.FloatTensor([torch.cos(theta), torch.sin(theta)])  # 2维标量,如tensor([ 0.1974, -0.9803])
    dists = torch.sum(coords * v, dim=-1)  # (7, 7) --> xcosθ+ysinθ

    # sigrange_line=[.25, 1.]-->   sig_line∈（0.25，1）
    sig_line = torch.rand(1)[0] * (sigrange_line[1] - sigrange_line[0]) + sigrange_line[0]
    # wmin_line=3 --> w_line∈(3,3.1)
    w_line = torch.rand(1)[0] * (0.5 * (N - 1) + 0.1 - wmin_line) + wmin_line

    vals_line = torch.exp(-dists ** 2 / 2. / sig_line ** 2) * (manhat < w_line)  # (7, 7)

    t = torch.rand(1)[0]
    vals = vals_nothing
    if t < (probs[0] + probs[1]):  # probs=[.25, .25]
        vals = vals_line
    else:
        vals = vals
    if t < probs[0]:
        vals = vals_gauss
    else:
        vals = vals

    v = vals / torch.sum(vals)  # 归一化 (7, 7)
    z = torch.zeros_like(v)
    f = torch.stack([v, z, z, z, v, z, z, z, v], dim=0).reshape([3, 3, N, N])
    return f


class BLUR(nn.Module):
    def __init__(self):
        super(BLUR, self).__init__()
        self.filter = random_blur_kernel
        self.probs = [.25, .25]
        self.sigrange_gauss = [1., 3.]
        self.sigrange_line = [.25, 1.]
        self.wmin_line = 3
        self.N_blur = 7  # k∈[3,7]的模糊核,k等于偶数时生成的图像全黑的，不知道为啥

    def forward(self, img):
        device = img.device
        f = self.filter(self.probs, self.N_blur, self.sigrange_gauss, self.sigrange_line, self.wmin_line).to(device)
        # print(f.device,img.d)
        out = F.conv2d(img, f, bias=None, padding=int((self.N_blur - 1) / 2))
        return out


if __name__ == '__main__':
    from PIL import Image, ImageOps

    img = Image.open('/opt/data/mingjin/pycharm/MyFirstNet/sample/origin/7.jpg')
    img = np.array(img) / 255.
    img_r = np.transpose(img, [2, 0, 1])
    img_tensor = torch.from_numpy(img_r).unsqueeze(0).float()
    B = BLUR()
    out = B(img_tensor)
    out = np.transpose(out.detach().squeeze(0).numpy(), [1, 2, 0])
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(out)
    plt.show()
