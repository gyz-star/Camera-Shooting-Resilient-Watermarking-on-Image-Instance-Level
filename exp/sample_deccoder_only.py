# !/usr/bin/env python
# -- coding: utf-8 --
# 小何一定行
import random
import sys

import PIL

sys.path.append("..")
from model import DeepHomographyModel, Encoder, Decoder, MaskRCNN
from dataset_deep_model import MyDataset
import matplotlib.pyplot as plt
from points_Ycbcr import template_embed_ycbcr as template_embed, template_extract_ycbcr as template_extract, genPN
from utils import get_cdos, get_warped_points_and_gap, get_rand_homography_mat, tensor2im, get_secret_acc, get_warped_extracted_IMG, im2tensor
from args import options
from noise_layers.compose import Compose
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
import torchvision
import kornia as K
import lpips
import torch.nn as nn

from noise_layers_exp.color import Color_Manipulation
from noise_layers_exp.blur import BLUR
from noise_layers_exp.jpeg import JPEGCompress
from noise_layers_exp.noise import Noise
from noise_layers_exp.resizeThenBack import RESIZE


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


def main(cp_path=None, cp_deep_path=None, img_path=None, dir_path=None, img_size=256):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=cp_path)
    parser.add_argument('--image', type=str, default=img_path)
    parser.add_argument('--images_dir', type=str, default=dir_path)
    # parser.add_argument('--save_dir', type=str, default=save_dir)
    args_this = parser.parse_args()

    if args_this.image is not None:
        files_list = [args_this.image]
    elif args_this.images_dir is not None:
        files_list = glob.glob(args_this.images_dir + '/*')
    else:
        print('Missing input image')
        return

    decoder = Decoder()
    rCnnNet = MaskRCNN()
    model_pth = torch.load(args_this.model, map_location=torch.device('cpu'))
    decoder.load_state_dict(model_pth['model_decoder'])
    decoder.eval()
    rCnnNet.eval()
    rCnnNet.to(device)
    decoder.to(device)

    deep_model = DeepHomographyModel().to(device)
    checkpoint = torch.load(cp_deep_path, map_location=torch.device('cpu'))
    deep_model.load_state_dict(checkpoint['model'])
    deep_model.eval()

    sec_true = [1., 0., 1., 1., 0., 1., 0., 1., 0., 0., 1., 1., 1., 0., 1.,
                1., 1., 1., 0., 1., 1., 1., 0., 1., 1., 1., 1., 0., 0., 0.]
    sec = torch.tensor(sec_true).unsqueeze(0).float().to(device)

    transform = transforms.Compose([
        # transforms.Resize((img_size, img_size), Image.BICUBIC),  # Image.BICUBIC,Image.NEAREST
        # transforms.Resize((266, 399), Image.BICUBIC),  # Image.BICUBIC,Image.NEAREST # hxw
        transforms.ToTensor(), ])

    with torch.no_grad():
        bit_SUM = 0
        COUNT = 0
        grid = []
        for filename in files_list:
            image = transform(Image.open(filename).convert('RGB')).unsqueeze(0).to(device)
            mask = rCnnNet(image)
            joint = image * mask

            f = torch.from_numpy(np.ones((3, 3, 11, 11), dtype=np.float32)).to(device)
            mask = F.conv2d(mask, f, bias=None, padding=3, dilation=2)
            mask = F.interpolate(mask, (img_size, img_size))
            mask[mask > 0] = 1.0
            # image = cv2.imread(filename)
            # h, w, _ = image.shape
            # print(h, w)
            # h_scale = h / 250
            # w_scale = w / 250
            # print((w // w_scale, h // h_scale))
            # image = cv2.resize(image, (int(w // w_scale), int(h // h_scale)))  # w*h
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2).float().to(device) / 255

            pred_bit, attention_map = decoder(joint)
            print(pred_bit)

            bit_acc = get_secret_acc(pred_bit, sec)
            bit_SUM += bit_acc
            COUNT += 1
            print('Count:%s\tbit_acc =  %.5f' % (COUNT, bit_acc))

            if dir_path is None:
                image_show = tensor2im(joint)
                plt.figure(1)
                plt.subplot(2, 4, 1)
                plt.axis('off')
                plt.imshow(image_show)
                plt.show()
            # save_name = filename.split('/')[-1].split('.')[0]

            # save_path_dec = args_this.save_dir + '/' + save_name + '_pp.jpg'
            # wait_dec = F.interpolate(wait_dec, (300, 300))
            # save_result = transforms.ToPILImage()(wait_dec.squeeze(0))
            # save_result.save(save_path_dec, quality=100)

        #     if COUNT >= 5 and len(grid) < 6:
        #         grid.append(dec_show)
        #         grid.append(attention_map[:, 0, ...].squeeze().cpu().numpy())
        #         # pred = im2tensor(palette[atten_show]).to(device)
        #         # grid.append(pred)

        # 打印PSNR,SSIM等信息
        print('img_nums:%s\tbit_acc =  %.5f\t' % (COUNT, bit_SUM / COUNT))


if __name__ == "__main__":
    img_size = 256
    device = torch.device("cuda:1")
    opt = options.getOpt()
    cp_path = '/opt/data/mingjin/pycharm/MyFirstNet/output/6_DDHNewAdjust5_Parameter/1138.pth'
    # cp_path = r'D:\BaiduNetdiskWorkspace\PycharmProject\Net\exp\1138.pth'  #
    # cp_deep_path = r'D:\BaiduNetdiskWorkspace\PycharmProject\Net\output\deep_model\AllNoise_1\500.pth'
    cp_deep_path = '/home/yizhi/output/deep_model/NoNoise_2_4points_addEncoder/300.pth'
    # img_path = None
    # img_path = '/home/yizhi/Net/duibi012shuiyin/17danquyuyouyang.jpg'
    # img_path = '/home/yizhi/Net/duibi012shuiyin/17shuangquyu.jpg'
    img_path = '/home/yizhi/Net/duibi012shuiyin/17danquyuzuoyang.jpg'
    # img_path = '/home/yizhi/Net/duibi012shuiyin/17ori.jpg'
    # img_path = '/home/yizhi/Net/twooriregion/23.jpg'
    # img_path = '/home/yizhi/Net/Heminpic/e.jpg'
    # img_path = '/home/yizhi/Net/image_folder/Wild/origin/e.jpg'
    # img_path = '/home/yizhi/bizhi/newbizhi/a19.jpg'
    # img_path = '/home/yizhi/image_folder/Wild/Ours/partial/aligned/screen_angle/left10/8.jpg'
    # img_path = '/home/yizhi/image_folder/Wild/demo/a123.jpg'
    # img_path = '/home/yizhi/image_folder/Wild/Ours/global/encoded/8.jpg'
    # img_path = '/home/yizhi/image_folder/Wild/Ours/global/screen_distance/10cm/IMG_20230601_204221.jpg'
    # img_path = r'D:\BaiduNetdisk  Workspace\PycharmProject\Net\image_folder\screen\screen_27_up2.jpg'
    # img_path = r'D:\BaiduNetdiskWorkspace\PycharmProject\Net\image_folder\screen\8_encoded.jpg'
    # img_path = '/home/yizhi/image_folder/Wild/Ours/partial/aligned/screen_angle/left60/6.jpg'
    dir_path = None
    # dir_path = '/home/yizhi/Net/singleregion'
    # dir_path = '/home/yizhi/Net/tworegion'
    # dir_path = '/home/yizhi/Net/orituregion'
    # dir_path = '/home/yizhi/image_folder/Wild/Ours/global/screen_angle/right60'
    # dir_path = '/opt/data/mingjin/pycharm/Net/image_folder/Wild/Ours/partial/captured/screen_distance/10cm'
    # dir_path = '/opt/data/mingjin/pycharm/Net/image_folder/Wild/Ours/partial/aligned/screen_distance/10cm'
    # dir_path = '/opt/data/mingjin/pycharm/Net/image_folder/Wild/Ours/partial/aligned/screen_angle/right10'
    # save_path = '/opt/data/mingjin/pycharm/Net/paper/c4/exp2-3/dec'
    main(cp_path, cp_deep_path, img_path, dir_path, img_size)
