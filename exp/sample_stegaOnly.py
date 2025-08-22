# !/usr/bin/env python
# -- coding: utf-8 --
# 小何一定行
import random
import sys
import time

import yaml
from easydict import EasyDict
from PIL import Image, ImageDraw, ImageFont

# from Net.exp.align_images import alignImage

sys.path.append("..")
from model import DeepHomographyModel, Encoder, Decoder, MaskRCNN
from dataset_deep_model import MyDataset
import matplotlib.pyplot as plt
# from points_embed import template_embed, template_extract
from utils import get_cdos, get_warped_points_and_gap, get_rand_homography_mat, tensor2im, get_secret_acc, get_warped_extracted_IMG, im2tensor, Gauss_noise, \
    Random_noise
from args import options
from noise_layers.compose import Compose
# from points_rgb import template_embed_rgb as template_embed, template_extract_rgb as template_extract, genPN
from points_Ycbcr import template_embed_ycbcr as template_embed, template_extract_ycbcr as template_extract, genPN
from noise import noise_layer
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
from noise_layers_exp.crop import Crop

from args import options

args = options.getOpt()


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


def main(cp_path=None, img_path=None, dir_path=None, save_dir=None, img_size=400):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=cp_path)
    parser.add_argument('--image', type=str, default=img_path)
    parser.add_argument('--images_dir', type=str, default=dir_path)
    parser.add_argument('--save_dir', type=str, default=save_dir)
    args_this = parser.parse_args()

    if args_this.save_dir is not None:
        if not os.path.exists(args_this.save_dir):
            os.makedirs(args_this.save_dir)

    if args_this.image is not None:
        files_list = [args_this.image]
    elif args_this.images_dir is not None:
        files_list = glob.glob(args_this.images_dir + '/*')
    else:
        print('Missing input image')
        return

    encoder = Encoder()
    decoder = Decoder()
    rCnnNet = MaskRCNN()
    model_pth = torch.load(args_this.model, map_location=torch.device('cpu'))
    encoder.load_state_dict(model_pth['model_encoder'])
    decoder.load_state_dict(model_pth['model_decoder'])
    encoder.eval()
    decoder.eval()
    rCnnNet.eval()
    rCnnNet.to(device)
    encoder.to(device)
    decoder.to(device)

    sec_true = [1., 0., 1., 1., 0., 1., 0., 1., 0., 0., 1., 1., 1., 0., 1.,
                1., 1., 1., 0., 1., 1., 1., 0., 1., 1., 1., 1., 0., 0., 0.]
    sec = torch.tensor(sec_true).unsqueeze(0).float().to(device)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), Image.BICUBIC),
        transforms.ToTensor(), ])

    with torch.no_grad():
        PSNR_SUM = 0
        LPIPS_SUM = 0
        SSIM_SUM = 0
        bit_SUM = 0
        ACE_SUM = 0
        COUNT = 0
        grid = []
        TIME_SUM = 0
        for filename in files_list:
            image = transform(Image.open(filename).convert('RGB')).unsqueeze(0).to(device)
            encoded = encoder(image, sec)
            se = torch.abs(image - encoded) * 5
            mask = rCnnNet(image)

            result = encoded * mask + image * (1 - mask)

            f = torch.from_numpy(np.ones((3, 3, 11, 11), dtype=np.float32)).to(device)
            mask = F.conv2d(mask, f, bias=None, padding=20, dilation=4)
            mask = F.interpolate(mask, (img_size, img_size))
            mask[mask > 0] = 1.0

            ###############################################################################

            # noised
            # compose = Compose(opt, None).to(device)
            # noised_img = compose(t_img, 100000)
            # ----------------------------------------------- #
            # kornia的噪声
            # wait_warp = t_img
            # wait_warp = result
            wait_warp = encoded

            # wait_warp = noise_layer(wait_warp, key='sample_deep').to(device)
            # resize = RESIZE()
            # # resize_scale = torch.Tensor(1).uniform_(0.9, 1.3).item()
            # resize_scale = 2
            # wait_warp = resize(wait_warp, resize_scale)

            crop = Crop(1)
            wait_warp = crop(wait_warp)

            # k_size_gauss = random.choice([3, 5, 7])
            # k_size_gauss = 5
            # wait_warp = K.filters.gaussian_blur2d(wait_warp, (k_size_gauss, k_size_gauss), (0.1, 1.0))
            #
            # k_size_motion = random.choice([3, 5, 7])
            # k_size_motion = 5
            # angle = torch.Tensor(wait_warp.size()[0]).uniform_(-35, 35)
            # direction = torch.from_numpy(np.random.choice([1, 0, -1], wait_warp.size()[0])).float()
            # wait_warp = K.filters.motion_blur(wait_warp, k_size_motion, angle, direction)
            #
            # contrast = [-0.2, 0.3]
            # color = [-0.05, 0.1]
            # brightness = [-0.2, 0.3]
            # contrast = [-0.1, 0.1]
            # color = [-0.03, 0.03]
            # brightness = [-0.1, 0.1]
            # con_scale = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
            # col_scale = torch.tensor(1.0).uniform_(color[0], color[1]).item()
            # bri_scale = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
            # wait_warp = (1 + con_scale) * (wait_warp + col_scale) + bri_scale
            # wait_warp = torch.clip(wait_warp, 0.0, 1.0)

            # random_noise_scale = torch.Tensor(1).uniform_(-0.03, 0.03).item()
            # gauss_noise_scale = torch.Tensor(1).uniform_(0., 0.02).item()
            #
            # wait_warp = Random_noise(wait_warp, random_noise_scale)
            # wait_warp = Gauss_noise(wait_warp, gauss_noise_scale)
            # wait_warp = torch.clip(wait_warp, 0.0, 1.0)

            # qf = int(100. - torch.rand(1)[0] * 50)
            # qf = 90
            # wait_warp = JPEGCompress(wait_warp, qf, device)
            # wait_warp = Color_Manipulation(args, wait_warp, 1000000)

            # wait_warp = torchgeometry.warp_perspective(wait_warp, homography[:, 1, :, :], dsize=(h, w))
            # warp_mask = torchgeometry.warp_perspective(mask, homography[:, 1, :, :], dsize=(h, w))
            # warp_mask[warp_mask > 0] = 1.0

            # grid = [result, wait_warp, torch.abs(result - wait_warp) * 10]
            # grid_show = torch.cat(grid, dim=0)
            # grid_save = torchvision.utils.make_grid(grid_show, nrow=1, padding=5)
            # save_name = filename.split('\\')[-1].split('.')[0] if filename.find('\\') != -1 else filename.split('/')[-1].split('.')[0]
            # path = save_dir + '/' + 'blur' + '.jpg'
            # torchvision.utils.save_image(grid_save, fp=path, nrow=1, padding=5)
            ###############################################################################

            wait_dec = wait_warp
            pred_bit, attention_map = decoder(wait_dec)
            # wait_mask = rCnnNet(wait_dec)
            # if wait_mask is None:
            #     continue
            # wait_mask = F.conv2d(wait_mask, f, bias=None, padding=20, dilation=4)
            # wait_mask = F.interpolate(wait_mask, (img_size, img_size))
            # wait_mask[wait_mask > 0] = 1.0

            # wait_mask = rectify_mask
            # print(wait_mask.shape, wait_dec.shape)
            # dec = get_warped_extracted_IMG(wait_dec.clone(), wait_mask.clone(), args=opt, isMul=False, isHomography=False, isResize=False)
            # pred_bit, attention_map = decoder(dec)

            bit_acc = get_secret_acc(pred_bit, sec)

            # 计算PSNR,SSIM
            PSNR = losses.psnr(encoded, image, 1.).item()
            SSIM = torch.mean(losses.ssim(encoded, image, 11)).item()
            loss_fn_alex = lpips.LPIPS(net='alex', verbose=False)
            loss_fn_alex = loss_fn_alex.to(device)
            LPIPS = torch.mean(loss_fn_alex(encoded, image)).item()

            # if bit_acc < 0.9:
            #     continue
            bit_SUM += bit_acc
            LPIPS_SUM += LPIPS
            PSNR_SUM += PSNR
            SSIM_SUM += SSIM
            COUNT += 1
            ACE = 0
            print('Count:%s\tPSNR = %.4f\tLPIPS = %.4f\tSSIM = %.4f\tbit_acc =  %.5f\tACE = %.4f' % (COUNT, PSNR, LPIPS, SSIM, bit_acc, ACE))

            # save_result = transforms.ToPILImage()(result.squeeze(0))
            # save_result.save('/opt/data/mingjin/pycharm/Net/image_folder/Ablation_deep/ours_result.jpg', quality=100)
            # save_wait_warp = transforms.ToPILImage()(wait_warp.squeeze(0))
            # save_wait_warp.save('/opt/data/mingjin/pycharm/Net/image_folder/Ablation_deep/ours_wait_warp.jpg', quality=100)
            #
            # save_rectify_img = transforms.ToPILImage()(rectify_img.squeeze(0))
            # save_rectify_img.save('/opt/data/mingjin/pycharm/Net/image_folder/Ablation_deep/ours_rectify_img.jpg', quality=100)
            #
            # imReg, H = alignImage(tensor2im(wait_warp), tensor2im(result))
            # rec_img = im2tensor(imReg).to(device)
            # save_rec_img = transforms.ToPILImage()(rec_img.squeeze(0))
            # save_rec_img.save('/opt/data/mingjin/pycharm/Net/image_folder/Ablation_deep/ours_rec_img.jpg', quality=100)

            if dir_path is None:
                image_show = tensor2im(image)
                encoded_show = tensor2im(encoded)
                mask_show = tensor2im(mask)
                se_show = tensor2im(se)
                result_show = tensor2im(result)
                noised_img_show = tensor2im(wait_warp)
                rectify_img_show = tensor2im(wait_dec)
                dec_show = tensor2im(dec)
                plt.figure(1)
                plt.subplot(2, 5, 1)
                plt.axis('off')
                plt.imshow(image_show)
                plt.subplot(2, 5, 2)
                plt.axis('off')
                plt.imshow(encoded_show)
                plt.subplot(2, 5, 3)
                plt.axis('off')
                plt.imshow(result_show)
                plt.subplot(2, 5, 4)
                plt.axis('off')
                plt.subplot(2, 5, 5)
                plt.axis('off')
                plt.imshow(mask_show)
                plt.subplot(2, 5, 6)
                plt.axis('off')
                plt.subplot(2, 5, 7)
                plt.axis('off')
                plt.subplot(2, 5, 8)
                plt.axis('off')
                plt.imshow(noised_img_show)
                plt.subplot(2, 5, 9)
                plt.axis('off')
                plt.imshow(rectify_img_show)
                plt.subplot(2, 5, 10)
                plt.axis('off')
                plt.imshow(dec_show)
                plt.show()

        # 打印PSNR,SSIM等信息
        print('img_nums:%s\tPSNR = %.3f\tLPIPS = %.3f\tSSIM = %.3f\tbit_acc =  %.5f\tACE =  %.5f' % (
            COUNT, PSNR_SUM / COUNT, LPIPS_SUM / COUNT, SSIM_SUM / COUNT, bit_SUM / COUNT, ACE_SUM / COUNT))
        print('time:%f' % (TIME_SUM / COUNT))


if __name__ == "__main__":
    img_size = 400
    device = torch.device("cuda:1")
    opt = options.getOpt()
    cp_deep_path = '/opt/data/mingjin/pycharm/Net/output/deep_model/NoNoise_2_4points_addEncoder/300.pth'
    cp_path = '/opt/data/mingjin/pycharm/MyFirstNet/output/6_DDHNewAdjust5_Parameter/1138.pth'
    # img_path = '/opt/data/mingjin/pycharm/Net/exp/sample/origin/17shuangquyu.jpg'
    # img_path = '/opt/data/mingjin/pycharm/Data/HiDDeN/val/000000004988.jpg'
    img_path = None
    # dir_path = None
    dir_path = '/opt/data/mingjin/pycharm/Data/HiDDeN/test'
    # dir_path = '/opt/data/mingjin/pycharm/MyFirstNet/sample/origin'
    # dir_path = '/opt/data/mingjin/pycharm/Data/HiDDeN/val'
    # dir_path = '/opt/data/mingjin/pycharm/Net/image_folder/Wild/origin'
    # save_path = '/opt/data/mingjin/pycharm/Net/sample/result/stega_demo'
    # save_path = '/opt/data/mingjin/pycharm/Net/paper/c4/exp1'
    save_path = '/opt/data/mingjin/pycharm/Net/image_folder/Wild/origin'
    # save_path = None
    main(cp_path, img_path, dir_path, save_path, img_size)
