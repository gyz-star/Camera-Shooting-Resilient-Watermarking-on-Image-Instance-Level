# !/usr/bin/env python
# -- coding: utf-8 --
# 小何一定行
import os
import random

import cv2
import numpy as np
import torchgeometry
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
from points_embed import genPN, template_embed, template_extract
from points_Ycbcr import genPN, template_embed_ycbcr, template_extract_ycbcr
from utils import tensor2im, get_warped_points_and_gap, get_rand_homography_mat, get_cdos, get_min_max_scale_extract


class MyDataset(Dataset):
    def __init__(self, path, image_size, message_length=30):
        super(MyDataset, self).__init__()
        self.img_path = []
        self.path = path
        # self.PN = PN
        self.img_size = (image_size, image_size)
        self.message_length = message_length
        self.transform = transforms.Compose([transforms.Resize(self.img_size),  # , Image.BICUBIC
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             ])

        # random.seed(42)
        files = os.listdir(path)
        for file in files:
            self.img_path.append(file)

    def __getitem__(self, idx):
        # img = cv2.imread(self.path + '/' + str(self.img_path[idx]), 1)  # 以RGB度图像读入
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # item_img = cv2.resize(img, self.img_size)  # (宽,高)

        img_path = self.path + '/' + str(self.img_path[idx])
        item_img = self.transform(Image.open(img_path).convert('RGB'))

        item_secret = np.random.binomial(1, 0.5, self.message_length)  # 秘密消息
        item_secret = torch.from_numpy(item_secret).float()

        return {'img': item_img, 'sec': item_secret}

        # return {'img': item_img}

    def __len__(self):
        return len(self.img_path)


if __name__ == "__main__":
    path = '/home/yizhi/Data/HiDDeN/train'
    # path = '/workshop/Datasets/COCO/train2017'  # 旧服务器
    image_expand_size = 256
    image_size = 256
    dataset = MyDataset(path, image_size=image_expand_size)
    dataloaderTrain = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=True)
    pn = genPN(7)

    print(len(dataloaderTrain))
    for i, batch in enumerate(dataloaderTrain):
        item_img = batch['img']  # 自然图
        h, w = item_img.shape[2:4]
        # item_img = batch['img'][0].unsqueeze(0) * 255
        # item_img = torch.ones_like(item_img)  # 全1图

        cdos = get_cdos(item_img)
        # print(cdos.shape)

        timg, PN = template_embed(item_img, cdos, PN=genPN(7), isOnes=False)
        # timg, PN = template_embed(item_img, cdos, PN=genPN(7), isOnes=True)

        homography = get_rand_homography_mat(timg, 0.2)
        homography = torch.from_numpy(homography).float()

        print(homography[:, 1, :, :].shape)
        warped_img = torchgeometry.warp_perspective(timg, homography[:, 1, :, :], dsize=(h, w))
        # warped_img = timg

        gap, warped_points = get_warped_points_and_gap(cdos, homography)
        print(gap[0])
        print(warped_points.shape)
        # wait_extract = warped_img[:, :, 22:278, 22:278]

        start = (image_expand_size - image_size) // 2
        end = start + image_size
        wait_extract = warped_img[:, :, start:end, start:end]

        pots = template_extract(wait_extract, PN)
        standard = template_extract(timg, PN)

        # print(torch.max(pots))
        print(warped_points[0])
        print(pots.shape)

        # img_show = tensor2im(item_img, is255=True)
        # timg_show = tensor2im(timg, is255=True)
        # pos_show = tensor2im(pots, is255=True)
        img_show = tensor2im(item_img[0].unsqueeze(0))
        source_points = np.array(cdos, np.int32).reshape((1, 9, 2))
        img_show = cv2.polylines(img_show.copy(), source_points, True, (255, 0, 0), 1, cv2.LINE_AA)
        timg_show = tensor2im(timg[0].unsqueeze(0))
        warped_show = tensor2im(warped_img[0].unsqueeze(0))
        wait_show = tensor2im(wait_extract[0].unsqueeze(0))
        # pos_show = tensor2im(pots)
        # img_show = item_img.long().cpu().squeeze().permute(1, 2, 0).numpy()
        # timg_show = timg.long().cpu().squeeze().permute(1, 2, 0).numpy()
        pow_ = torch.cat([pots, pots, pots], dim=1)
        pos_show = tensor2im(pow_[0].unsqueeze(0))

        standard_ = torch.cat([standard, standard, standard], dim=1)
        standard_show = tensor2im(standard_[0].unsqueeze(0))
        standard_show = cv2.polylines(standard_show.copy(), source_points, True, (255, 0, 0), 1, cv2.LINE_AA)
        print(pos_show.shape)

        end_points = np.array(cdos + gap[0].cpu().detach().numpy(), np.int32).reshape((1, 9, 2))
        print(end_points)
        pos_show = cv2.polylines(pos_show.copy(), end_points, True, (255, 0, 0), 1, cv2.LINE_AA)
        # pos_show = pots[0].detach().squeeze().cpu().numpy()
        # pos_show = (get_min_max_scale_extract(pots))[0].detach().squeeze().cpu().numpy()
        # pos_show = min_max_scale(pots[0].detach().squeeze().cpu().numpy())
        # print(timg_show.shape)
        # print(img_show.shape)
        # print(pos_show.shape)

        # print(pos_show[10][10])

        plt.figure(1)
        plt.subplot(2, 3, 1)
        plt.imshow(img_show)
        plt.subplot(2, 3, 2)
        plt.imshow(timg_show)
        plt.subplot(2, 3, 3)
        plt.imshow(warped_show)
        plt.subplot(2, 3, 4)
        plt.imshow(wait_show)
        plt.subplot(2, 3, 5)
        # plt.imshow(pos_show, cmap=plt.cm.gray)
        plt.imshow(pos_show)
        plt.subplot(2, 3, 6)
        plt.imshow(standard_show)
        plt.show()
        break
