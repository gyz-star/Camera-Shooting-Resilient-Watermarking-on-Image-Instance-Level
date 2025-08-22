# !/usr/bin/env python
# -- coding: utf-8 --
# 小何一定行
import datetime
import random

import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F


def choice(img, N, pad_pix, rand_num_pos):
    h, w = img.shape[2:4]
    points = []
    dx = int(w / (N - 1))
    for i in range(N):
        points.append((dx * i, pad_pix))
        points.append((dx * i, -pad_pix + h))

    # 原点
    source = np.array(points, np.int32)
    source = source.reshape(1, -1, 2)

    # 随机扰动幅度
    # rand_num_pos = random.uniform(20, 30)
    rand_num_neg = -1 * rand_num_pos

    newpoints = []
    for i in range(N):
        rand = np.random.choice([rand_num_neg, rand_num_pos], p=[0.5, 0.5])
        if i == 1:
            nx_up = points[2 * i][0]
            ny_up = points[2 * i][1] + rand
            nx_down = points[2 * i + 1][0]
            ny_down = points[2 * i + 1][1] + rand
        elif i == 4:
            rand = rand_num_neg if rand > 1 else rand_num_pos
            nx_up = points[2 * i][0]
            ny_up = points[2 * i][1] + rand
            nx_down = points[2 * i + 1][0]
            ny_down = points[2 * i + 1][1] + rand
        else:
            nx_up = points[2 * i][0]
            ny_up = points[2 * i][1]
            nx_down = points[2 * i + 1][0]
            ny_down = points[2 * i + 1][1]

        newpoints.append((nx_up, ny_up))
        newpoints.append((nx_down, ny_down))

    # target点
    target = np.array(newpoints, np.int32)
    target = target.reshape(1, -1, 2)
    # 计算matches
    matches = []
    for i in range(1, 2 * N + 1):
        matches.append(cv2.DMatch(i, i, 0))
    return source, target, matches


def norm(points_int, width, height):
    """
    将像素点坐标归一化至 -1 ~ 1
    """
    points_int_clone = torch.from_numpy(points_int).detach().float()
    x = ((points_int_clone * 2)[..., 0] / (width - 1) - 1)
    y = ((points_int_clone * 2)[..., 1] / (height - 1) - 1)
    return torch.stack([x, y], dim=-1).contiguous().view(-1, 2)


class TPS(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, img, mask, N, pad_pix, rand_num_pos):
        source, target, matches = choice(img, N, pad_pix, rand_num_pos)
        h, w = img.shape[2:4]
        device = img.device
        ten_source = norm(source, w, h)
        ten_target = norm(target, w, h)
        X, Y = ten_target[None, ...].to(device), ten_source[None, ...].to(device)  # ten_target[None, ...]相当于tensor_target.unsqueeze(0)
        """ 计算grid"""

        grid = torch.ones(1, h, w, 2, device=device)
        grid[:, :, :, 0] = torch.linspace(-1, 1, w)
        grid[:, :, :, 1] = torch.linspace(-1, 1, h)[..., None]
        grid = grid.view(-1, h * w, 2)

        """ 计算W, A"""
        n, k = X.shape[:2]
        Z = torch.zeros(1, k + 3, 2, device=device)
        P = torch.ones(n, k, 3, device=device)
        L = torch.zeros(n, k + 3, k + 3, device=device)

        eps = 1e-9
        D2 = torch.pow(X[:, :, None, :] - X[:, None, :, :], 2).sum(-1).to(device)
        K = D2 * torch.log(D2 + eps)

        P[:, :, 1:] = X
        Z[:, :k, :] = Y
        L[:, :k, :k] = K
        L[:, :k, k:] = P
        L[:, k:, :k] = P.permute(0, 2, 1)

        Q = torch.solve(Z, L)[0]
        W, A = Q[:, :k], Q[:, k:]

        """ 计算U """
        eps = 1e-9
        D2 = torch.pow(grid[:, :, None, :] - X[:, None, :, :], 2).sum(-1)
        U = D2 * torch.log(D2 + eps)

        """ 计算P """
        n, k = grid.shape[:2]
        device = grid.device
        P = torch.ones(n, k, 3, device=device)
        P[:, :, 1:] = grid

        # grid = P @ A + U @ W
        grid = torch.matmul(P, A) + torch.matmul(U, W)

        warped_grid = grid.view(-1, h, w, 2)
        batch_size = img.shape[0]
        warped_grid = warped_grid.expand(batch_size, -1, -1, -1)
        # warp_img = torch.grid_sampler_2d(img, warped_grid, 0, 0, align_corners=True)
        warp_img = F.grid_sample(img, warped_grid, align_corners=True)
        warp_mask = F.grid_sample(mask, warped_grid, align_corners=True)
        warp_mask[warp_mask > 0] = 1
        return warp_img, warp_mask


if __name__ == '__main__':
    import sys
    from PIL import Image
    # import yaml
    # from easydict import EasyDict
    # from torch.utils.data import DataLoader
    # import torchvision.utils as vutils
    #
    # sys.path.append('..')
    # from data import STN_dataset, EnDe_dataset, EnDeMask_dataset
    # from tensorboardX import SummaryWriter
    #
    # # log = './log/'
    # # writer = SummaryWriter(logdir=log, comment='_scalars')
    #
    # with open('../args/endemask_options.yaml', 'r') as f:
    #     args = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))
    # dataset = EnDeMask_dataset.myDataset(args)
    # print(len(dataset))
    # bz = 12
    # dataloader = DataLoader(dataset, batch_size=bz, shuffle=True)
    # img, mask, sec = next(iter(dataloader))
    # device = torch.device('cuda:1')
    # img = img.to(device)
    # mask = mask.to(device)
    #
    # time_star = datetime.datetime.now()
    # time_star_str = str(time_star)
    #
    # tps = TPS()
    # tps.to(device)
    # warp_img, warp_mask = tps(img, mask, 5, 30, 30)
    #
    # print(warp_img.shape, warp_img.device)
    #
    # time_end = datetime.datetime.now()
    # time_end_str = str(time_end)
    #
    # print('time_star:', time_star_str)
    # print('time_end:', time_end_str)
    #
    # ori_img = np.array(ToPILImage()((img[0]).cpu()))
    # new_img_torch = np.array(ToPILImage()((warp_img[0]).cpu()))
    # ori_mask = np.array(ToPILImage()((mask[0]).cpu()))
    # new_mask_torch = np.array(ToPILImage()((warp_mask[0]).cpu()))
    # print(np.unique(new_mask_torch))
    # import matplotlib.pyplot as plt
    #
    # plt.subplot(2, 2, 1)
    # plt.imshow(ori_img)
    # plt.subplot(2, 2, 2)
    # plt.imshow(new_img_torch)
    # plt.subplot(2, 2, 3)
    # plt.imshow(ori_mask)
    # plt.subplot(2, 2, 4)
    # plt.imshow(new_mask_torch)
    # plt.show()

    # with torch.no_grad():
    #     origin_grid = torchvision.utils.make_grid(img, nrow=bz, normalize=True, scale_each=True, padding=2)
    #     warp_grid = torchvision.utils.make_grid(warp_img, nrow=bz, normalize=True, scale_each=True, padding=2)
    #     writer.add_image("image_input", origin_grid, 0)
    #     writer.add_image("image_warp", warp_grid, 0)
    # writer.close()
    def get_tps_param(args, global_step):
        ramp_fn = lambda ramp: np.min([global_step / ramp, 1.])  # [1/1000，1]
        # degree_warp = torch.rand(1)[0] * ramp_fn(args.tps_ramp) * args.degree_tps
        degree_warp = 30
        keypoint_position = random.choice([10, 20, 30, 40, 50, 60])
        keypoint_nums = random.randint(5, 20)
        return degree_warp, keypoint_position, keypoint_nums

    img_path = '/opt/data/mingjin/pycharm/MyFirstNet/sample/origin/1.jpg'
    img = transforms.ToTensor()(transforms.Resize((256,256))(Image.open(img_path).convert('RGB')))
    # print(img.shape)
    choice(img,10,)
