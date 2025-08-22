# !/usr/bin/env python
# -- coding: utf-8 --
# 小何一定行
import random
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import cv2
import torch.nn.functional as F
import numpy as np
import torch, math
import torch.nn as nn
import torchgeometry
from matplotlib import pyplot as plt
import yaml
from easydict import EasyDict
import sys

sys.path.append('..')
from noise_layers.color import Color_Manipulation
from noise_layers.blur import BLUR
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.jpeg import JPEGCompress
# from noise_layers_exp.jpegUDH import JpegCompression2
from noise_layers.noise import Noise
from noise_layers.resizeThenBack import Resize


##################
#  5种良性噪声组合 #
##################

class Compose(nn.Module):
    def __init__(self, args, writer=None):
        super(Compose, self).__init__()
        self.device = torch.device(args.device_ID)
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.writer = writer
        self.blur = BLUR()
        self.color = Color_Manipulation
        self.noise = Noise
        self.resize = Resize
        # self.jpeg = JpegCompression(self.device)  # UDH里的JPEG压缩
        # self.jpeg = JpegCompression2(args, self.args.jpeg_quality)  # UDH改进型jpeg
        self.jpeg = JPEGCompress  # StegaStamp里的JPEG压缩

    def forward(self, img, glob_step):
        # batch = img.shape[0]
        # group = batch // 2
        # # print(group)
        # a = self.blur(img[0:group, ...].clone())
        # a = self.color(self.args, a, glob_step, self.writer)
        # a = self.noise(self.args, a, glob_step, self.writer)
        #
        # b = self.jpeg(self.args, img[group:batch, :, :, :].clone(), glob_step, self.writer)
        # # b = self.jpeg(img[group:batch, ...].clone())
        # # b = self.jpeg(img[group:batch, ...].clone(), glob_step, self.writer)
        # result = torch.cat([a, b], dim=0)
        # return result
        # out = img
        out = self.resize(self.args, img, glob_step, self.writer)
        out = self.blur(out)
        out = self.color(self.args, out, glob_step, self.writer)
        out = self.noise(self.args, out, glob_step, self.writer)
        out = self.jpeg(self.args, out, glob_step, self.writer)
        # out = self.jpeg(out, glob_step, self.writer, train_type='train')
        # out = self.jpeg(out)
        return out


if __name__ == '__main__':
    sys.path.append("..")
    from PIL import Image, ImageOps
    from matplotlib import pyplot as plt
    import yaml
    from torchvision import transforms
    from easydict import EasyDict
    from utils import get_warped_extracted_IMG, tensor2im, extractIMG, get_rand_homography_mat
    from args import options
    from model import DeepHomographyModel, Encoder, Decoder, MaskRCNN

    img_size = 256
    device = torch.device("cuda:3")


    def Gauss_noise(inputs, noise_factor=0.15):
        noisy = inputs + torch.randn_like(inputs) * noise_factor
        noisy = torch.clip(noisy, 0., 1.)
        return noisy


    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), Image.BICUBIC),
        transforms.ToTensor(), ])

    opt = options.getOpt()
    rCnnNet = MaskRCNN()
    rCnnNet.eval()
    rCnnNet.to(device)
    compose = Compose(opt)
    glob_step = 100000
    img_path = '/opt/data/mingjin/pycharm/MyFirstNet/sample/origin/3.jpg'
    # image_path = '/opt/data/mingjin/pycharm/Net/noise_layers_exp/demo/7_jpeg.jpg'

    # mask_path = '/opt/data/mingjin/pycharm/Net/sample/result/stega_demo/7_mask.jpg'

    image = transform(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)
    # mask = transform(Image.open(mask_path).convert('RGB')).unsqueeze(0).to(device)
    # mask[mask > 0] = 1
    noised = compose(image, glob_step).to(device)

    h, w = image.shape[2:4]
    homography = get_rand_homography_mat(image, 0.2)
    homography = torch.from_numpy(homography).float().to(device)
    noised = torchgeometry.warp_perspective(noised, homography[:, 1, :, :], dsize=(h, w))

    # out = Gauss_noise(image, 0.1)
    # out = image
    mask_img = rCnnNet(image)
    print(mask_img.shape)
    mask_noised = rCnnNet(noised)
    print(mask_noised.shape)

    # dec = extractIMG(image, mask, args=opt, isMul=False, isHomography=True, isResize=False, isBack=False)

    img_show = tensor2im(image)
    noised_show = tensor2im(noised)
    mask_img_show = tensor2im(mask_img)
    mask_noised_show = tensor2im(mask_noised)
    # dec_show = tensor2im(dec)
    plt.figure(1)
    plt.subplot(2, 2, 1)
    plt.axis('off')
    plt.imshow(img_show)
    plt.subplot(2, 2, 2)
    plt.axis('off')
    plt.imshow(noised_show)
    plt.subplot(2, 2, 3)
    plt.axis('off')
    plt.imshow(mask_img_show)
    plt.subplot(2, 2, 4)
    plt.axis('off')
    plt.imshow(mask_noised_show)
    plt.show()

    save_dir = './demo/'
    save_name = img_path.split('/')[-1].split('.')[0]
    save_path_image = save_dir + '/' + save_name + '_image.jpg'
    save_path_result = save_dir + '/' + save_name + '_jpeg.jpg'
    save_path_dec = save_dir + '/' + save_name + '_dec.jpg'
    image_show = cv2.cvtColor(img_show, cv2.COLOR_RGB2BGR)
    result_show = cv2.cvtColor(noised_show, cv2.COLOR_RGB2BGR)
    # dec_show = cv2.cvtColor(dec_show, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(save_path_image, image_show, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    # cv2.imwrite(save_path_dec, dec_show, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    # cv2.imwrite(save_path_result, result_show, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
