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


def main(cp_path=None, cp_deep_path=None, img_path=None, dir_path=None, img_size=256, p_nums=4):
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

    deep_model = DeepHomographyModel(p_nums).to(device)
    checkpoint = torch.load(cp_deep_path, map_location=torch.device('cpu'))
    # pn = checkpoint['pn_embed']
    deep_model.load_state_dict(checkpoint['model'])
    deep_model.eval()

    sec_true = [1., 0., 1., 1., 0., 1., 0., 1., 0., 0., 1., 1., 1., 0., 1.,
                1., 1., 1., 0., 1., 1., 1., 0., 1., 1., 1., 1., 0., 0., 0.]
    sec = torch.tensor(sec_true).unsqueeze(0).float().to(device)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), Image.BICUBIC),  # Image.BICUBIC,Image.NEAREST
        transforms.ToTensor(), ])

    with torch.no_grad():
        bit_SUM = 0
        COUNT = 0
        Mistake_Count = 0
        grid = []
        for filename in files_list:
            image = transform(Image.open(filename).convert('RGB')).unsqueeze(0).to(device)

            # image = cv2.imread(filename)
            # image = cv2.resize(image, (256, 256), cv2.INTER_NEAREST)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = torch.from_numpy(image.transpose((2, 0, 1))).float().div(255).unsqueeze(0).to(device)

            # ###############################################################################
            pn = genPN(7)

            item_ones = torch.ones_like(image).float().to(device)

            # homography = get_rand_homography_mat(image, 0.2)
            # homography = torch.from_numpy(homography).float().to(device)
            # # homography = get_rand_homography(image, 25, 25)
            # # homography = torch.from_numpy(homography).float().to(device)
            #
            source_points = get_cdos(image, p_nums)

            t_ones, PN2 = template_embed(item_ones, source_points, PN=pn)

            # #####################################################
            Ip1 = template_extract(image, PN2).to(device)
            Ip2 = template_extract(t_ones, PN2).to(device)

            inputs = torch.cat([Ip1, Ip2], dim=1)
            predict_gap = deep_model.forward(inputs)
            predict_points = source_points + predict_gap.view(-1, p_nums, 2).detach().cpu().numpy()

            h, w = image.shape[2:4]
            m, _ = cv2.findHomography(np.float32(source_points).reshape(-1, 2), np.float32(predict_points).reshape(-1, 2))
            m_inverse = np.linalg.inv(m)
            m_inverse = torch.from_numpy(m_inverse).unsqueeze(0).float().to(device)
            rectify_img = torchgeometry.warp_perspective(image, m_inverse, dsize=(h, w))

            mask = rCnnNet(rectify_img)
            if mask is None:
                continue
            f = torch.from_numpy(np.ones((3, 3, 11, 11), dtype=np.float32)).to(device)
            rectify_mask = F.conv2d(mask, f, bias=None, padding=20, dilation=4)
            rectify_mask = F.interpolate(rectify_mask, (img_size, img_size))
            rectify_mask[rectify_mask > 0] = 1.0

            # rectify_mask = K.geometry.warp_perspective(mask, m_inverse, dsize=(h, w))  # mode='nearest', align_corners=True
            # rectify_mask[rectify_mask > 0] = 1
            # ###############################################################################

            # wait_dec = image
            # wait_mask = mask
            # wait_dec = noised_img
            wait_dec = rectify_img
            wait_mask = rectify_mask
            wait_dec = get_warped_extracted_IMG(wait_dec.clone(), wait_mask.clone(), args=opt, isMul=False, isHomography=False, isResize=False)
            pred_bit, attention_map = decoder(wait_dec)

            bit_acc = get_secret_acc(pred_bit, sec)

            if bit_acc <= 0.6:
                Mistake_Count += 1
                continue
            bit_SUM += bit_acc
            COUNT += 1
            print('Count:%s\tbit_acc =  %.5f' % (COUNT, bit_acc))

            image_show = tensor2im(image)
            mask_show = tensor2im(mask)
            rectify_img_show = tensor2im(rectify_img)
            rectify_mask_show = tensor2im(rectify_mask)
            dec_show = tensor2im(wait_dec)
            Ip1_show = tensor2im(Ip1)
            Ip2_show = tensor2im(Ip2)

            if dir_path is None:
                plt.figure(1)
                plt.subplot(2, 4, 1)
                plt.axis('off')
                plt.imshow(image_show)
                plt.subplot(2, 4, 2)
                plt.axis('off')
                plt.imshow(rectify_img_show)
                plt.subplot(2, 4, 3)
                plt.axis('off')
                plt.imshow(mask_show)
                plt.subplot(2, 4, 4)
                plt.axis('off')
                plt.imshow(rectify_mask_show)
                plt.subplot(2, 4, 5)
                plt.axis('off')
                plt.subplot(2, 4, 6)
                plt.axis('off')
                plt.imshow(dec_show)
                plt.subplot(2, 4, 7)
                plt.axis('off')
                plt.imshow(Ip1_show, cmap='gray')
                plt.subplot(2, 4, 8)
                plt.axis('off')
                plt.imshow(Ip2_show, cmap=plt.cm.gray)
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
        print('img_nums:%s\tbit_acc =  %.5f\tmistake_count = %s' % (COUNT, bit_SUM / COUNT, Mistake_Count))


if __name__ == "__main__":
    img_size = 400
    device = torch.device("cpu")
    opt = options.getOpt()
    # cp_deep_path = '/opt/data/mingjin/pycharm/Net/output/Vanilla/model_final_end501_Warp0.2_Vanilla.pth'
    # cp_deep_path = '/opt/data/mingjin/pycharm/Net/output/Vanilla/model_final_end501_Warp0.2AndAllNoise_version3_Vanilla.pth'
    # cp_deep_path = '/opt/data/mingjin/pycharm/Net/output/Vanilla/model_final_end501_Warp0.2AndAllNoise_beginVersion3Continue500_version3Vanilla.pth'
    # cp_path = '/opt/data/mingjin/pycharm/MyFirstNet/output/6_DDHNewAdjust5_Parameter/1139.pth'
    # cp_path = '/opt/data/mingjin/pycharm/Net/output/retrain the previous model_1/1255.pth'
    # cp_path = r'D:\BaiduNetdiskWorkspace\PycharmProject\Net\exp\1139.pth'  # 方案2
    # cp_path = r'D:\BaiduNetdiskWorkspace\PycharmProject\Net\exp\1138.pth'  #
    # cp_deep_path = r'D:\BaiduNetdiskWorkspace\PycharmProject\Net\exp\100.pth'
    # cp_deep_path = r'D:\BaiduNetdiskWorkspace\PycharmProject\Net\output\deep_model\AllNoise_1\500.pth'
    cp_path = '/opt/data/mingjin/pycharm/MyFirstNet/output/6_DDHNewAdjust5_Parameter/1138.pth'  # 方案2
    cp_deep_path = '/home/yizhi/output/deep_model/NoNoise_2_4points_addEncoder/300.pth'
    # img_path = '/home/yizhi/image_folder/Wild/Ours/global/encoded/8.jpg'
    # img_path = '/home/yizhi/image_folder/Wild/Ours/global/screen_angle/left10/IMG_20230605_113736.jpg'
    # img_path = None
    # img_path = '/home/yizhi/bizhi/newbizhi/19.jpg'
    # img_path = '/home/yizhi/bizhi/sheep30cm.jpg'
    # img_path = '/home/yizhi/image_folder/Wild/Ours/partial/encoded/8.jpg'


    # img_path = '/home/yizhi/image_folder/Wild/Ours/global/screen_distance/30cm/IMG_20230601_210745.jpg'#拍摄距离30cm无角度
    img_path = '/home/yizhi/image_folder/Wild/Ours/global/screen_distance/50cm/IMG_20230601_215119.jpg'#拍摄距离50cm无角度
    # img_path = '/home/yizhi/image_folder/Wild/Ours/global/screen_angle/left10/IMG_20230605_113905.jpg'


    # img_path = '/home/yizhi/image_folder/Wild/demo/a10.jpg'
    # img_path = '/home/yizhi/image_folder/Wild/print/print/10-2.jpg'
    # img_path = r'D:\BaiduNetdiskWorkspace\PycharmProject\Net\image_folder\screen\screen_3_down1.jpg'
    # img_path = r'D:\BaiduNetdiskWorkspace\PycharmProject\Net\image_folder\screen\8_encoded.jpg'
    dir_path = None
    # dir_path = '/home/yizhi/image_folder/Wild/Ours/global/screen_distance/30cm'
    # dir_path = '/opt/data/mingjin/pycharm/Net/image_folder/Wild/print/ours'
    # save_path = '/opt/data/mingjin/pycharm/Net/paper/c4/exp2-3/dec'
    main(cp_path, cp_deep_path, img_path, dir_path, img_size, 4)
