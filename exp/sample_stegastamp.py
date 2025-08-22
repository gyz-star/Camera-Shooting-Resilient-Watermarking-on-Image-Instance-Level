#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 小何一定行
# import argparse
#
# # required参数可以设置该参数是否必需。
# parser = argparse.ArgumentParser(description='姓名')
# parser.add_argument('--family', type=str, default='张',help='姓')
# parser.add_argument('--name', type=str, default='三', help='名')
# args = parser.parse_args()
#
# #打印姓名
# print(args.family+args.name)
import glob
import os
import random
import sys
import kornia as K
import cv2
import torchgeometry
import torchvision
import torch.nn.functional as F
from Net.noise_layers.resizeThenBack import RESIZE
from Net.utils import Gauss_noise, get_secret_acc, get_rand_homography_mat

sys.path.append('')
import torch
import matplotlib.pyplot as plt
import numpy as np
from stegastamp_dir import model, dataset
from torchvision import transforms
from PIL import Image, ImageOps
from kornia import color, losses
from stegastamp_dir.dataset import StegaData
from torch.utils.data import DataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device = torch.device('cuda:0')


def get_rand_homography(img, eps_x, eps_y):
    res = np.zeros((1, 2, 3, 3))

    h, w = img.shape[2:4]

    top_left_x = random.uniform(-eps_x, eps_x)
    top_left_y = random.uniform(-eps_y, eps_y)
    bottom_left_x = random.uniform(-eps_x, eps_x)
    bottom_left_y = random.uniform(-eps_y, eps_y)
    top_right_x = random.uniform(-eps_x, eps_x)
    top_right_y = random.uniform(-eps_y, eps_y)
    bottom_right_x = random.uniform(-eps_x, eps_x)
    bottom_right_y = random.uniform(-eps_y, eps_y)

    rect = np.array([
        [top_left_x, top_left_y],
        [top_right_x + w, top_right_y],
        [bottom_right_x + w, bottom_right_y + h],
        [bottom_left_x, bottom_left_y + h]], dtype="float32")

    dst = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]], dtype="float32")

    res_i = cv2.getPerspectiveTransform(rect, dst)
    res_i_inv = np.linalg.inv(res_i)

    res[0, 0] = res_i
    res[0, 1] = res_i_inv
    return res


# checkpoint = torch.load(
#     '/opt/data/mingjin/pycharm/Exercise/stegastamp/StegaStamp-pytorch/result_model/result_model_StegaStamp.pth')
checkpoint = torch.load(
    '/opt/data/mingjin/pycharm/Exercise/stegastamp/StegaStamp-pytorch/checkpoints/EXP_NAME=6/120000.pth')

encoder = model.StegaStampEncoder()
decoder = model.StegaStampDecoder(secret_size=100)
if torch.cuda.is_available():
    encoder = encoder.to(device)
    decoder = decoder.to(device)

encoder.load_state_dict(checkpoint['model_encoder'])
decoder.load_state_dict(checkpoint['model_decoder'])

# train_path = r'/opt/data/mingjin/pycharm/MyFirstNet/sample/origin'
train_path = r'/home/yizhi/exp/sample/c4'
dataset = StegaData(train_path)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

files_list = glob.glob(train_path + '/*')
transform = transforms.Compose([
    transforms.Resize((256, 256), Image.BICUBIC),
    transforms.ToTensor(), ])

# secret = np.random.binomial(1, 0.5, 100)
# secret = torch.from_numpy(secret).float().to(device)

# img_cover_path = r'/home/mingjin/pycharm/Exercise/StegaStamp/StegaStamp-pytorch/valid/4.jpeg'
# img_cover = Image.open(img_cover_path).convert('RGB')
# # ImageOps.fit()图像被裁剪到指定的宽高比和尺寸
# size = (400, 400)
# img_cover = ImageOps.fit(img_cover, size)
# img_cover = transforms.ToTensor()(img_cover).unsqueeze(0).to(device)
# img_cover, secret = next(iter(dataloader))
grid = []
# for i, batch in enumerate(dataloader):
for filename in files_list:
    secret = np.random.binomial(1, 0.5, 100)
    secret = torch.from_numpy(secret).float()
    img_cover = transform(Image.open(filename).convert('RGB')).unsqueeze(0).to(device)
    # img_cover, secret = batch

    img_cover = F.interpolate(img_cover, size=(400, 400), mode='bilinear', align_corners=True)

    secret, img_cover = secret.to(device), img_cover.to(device)
    inputs = (secret, img_cover)

    residual = encoder(inputs)
    result_img = torch.clamp(img_cover + residual, 0, 1)
    print(losses.psnr(result_img, img_cover, 1.).item())
    print(torch.mean(losses.ssim(result_img, img_cover, 11)).item())

    wait_warp = result_img
    # resize = RESIZE()
    # wait_warp = resize(wait_warp, 2.5)
    # wait_warp = K.enhance.adjust_brightness(wait_warp, 0.0092)
    # wait_warp = K.enhance.adjust_hue(wait_warp, -10/255)
    # wait_warp = K.enhance.adjust_contrast(wait_warp, 0.6)
    # wait_warp = K.filters.gaussian_blur2d(wait_warp, (7, 7), (0.1, 1.0))
    # wait_warp = K.filters.motion_blur(wait_warp, 7, 15., 1)
    # wait_warp = JPEGCompress(wait_warp, 90, device)
    # wait_warp = Gauss_noise(wait_warp, 0.02)

    # h, w = wait_warp.shape[2:4]
    # homography = get_rand_homography(wait_warp, 20, 20)
    # homography = torch.from_numpy(homography).float().to(device)
    # wait_warp = torchgeometry.warp_perspective(wait_warp, homography[:, 1, :, :], dsize=(h, w))

    # result_sec = torch.round(decoder(wait_warp))
    # bit_acc = get_secret_acc(secret, result_sec)
    # print(bit_acc)

    result_img = F.interpolate(img_cover, size=(256, 256), mode='bilinear', align_corners=True)
    grid.append(result_img)
grid_show = torch.cat(grid, dim=0)
grid_save = torchvision.utils.make_grid(grid_show, nrow=1, padding=5)
path = '/opt/data/mingjin/pycharm/Net/paper/c4/exp3/origin.png'
torchvision.utils.save_image(grid_save, fp=path, nrow=1, padding=5)
