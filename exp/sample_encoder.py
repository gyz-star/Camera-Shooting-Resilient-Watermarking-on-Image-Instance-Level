# !/usr/bin/env python
# -- coding: utf-8 --
# 小何一定行
import random
import sys

import kornia

sys.path.append("..")
from model import DeepHomographyModel, Encoder, Decoder, MaskRCNN
from dataset_deep_model import MyDataset
import matplotlib.pyplot as plt
from utils import get_cdos, get_warped_points_and_gap, get_rand_homography_mat, tensor2im, get_secret_acc, get_warped_extracted_IMG, im2tensor
from args import options
from noise_layers.compose import Compose
from points_Ycbcr import template_embed_ycbcr as template_embed, template_extract_ycbcr as template_extract, genPN
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


def main(cp_path=None, img_path=None, dir_path=None, save_dir=None, img_size=256, p_num=4):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=cp_path)
    parser.add_argument('--image', type=str, default=img_path)
    parser.add_argument('--images_dir', type=str, default=dir_path)
    parser.add_argument('--save_dir', type=str, default=save_dir)
    args_this = parser.parse_args()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args_this.image is not None:
        files_list = [args_this.image]
    elif args_this.images_dir is not None:
        files_list = glob.glob(args_this.images_dir + '/*')
    else:
        print('Missing input image')
        return

    encoder = Encoder()
    rCnnNet = MaskRCNN()
    model_pth = torch.load(args_this.model, map_location=torch.device('cpu'))
    encoder.load_state_dict(model_pth['model_encoder'])
    encoder.eval()
    rCnnNet.eval()
    rCnnNet.to(device)
    encoder.to(device)

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
        COUNT = 0
        grid = []
        # print(files_list)
        for i, filename in enumerate(files_list):
            # print(i)
            image = transform(Image.open(filename).convert('RGB')).unsqueeze(0).to(device)
            encoded = encoder(image, sec)
            se = torch.abs(image - encoded) * 5
            mask = rCnnNet(image)
            joint = encoded * mask + image * (1 - mask)

            f = torch.from_numpy(np.ones((3, 3, 11, 11), dtype=np.float32)).to(device)
            mask = F.conv2d(mask, f, bias=None, padding=3, dilation=2)
            mask = F.interpolate(mask, (img_size, img_size))
            mask[mask > 0] = 1.0

            # se = encoded * (mask)

            source_points = get_cdos(joint, p_num)
            pn = genPN(7)
            result, PN1 = template_embed(joint, source_points, PN=pn)

            rere=torch.abs(image - encoded) * 5
            # result = F.interpolate(joint, size=(512, 512), mode='nearest')
            # mask = F.interpolate(mask, size=(512, 512), mode='nearest')
            # mask[mask > 0] = 1
            dec = get_warped_extracted_IMG(result.clone(), mask.clone(), args=opt, isMul=False, isHomography=False, isResize=False)

            # 计算PSNR,SSIM
            PSNR = losses.psnr(result, image ,1.).item()
            SSIM = torch.mean(losses.ssim(result, image,11)).item()
            loss_fn_alex = lpips.LPIPS(net='alex', verbose=False)
            loss_fn_alex = loss_fn_alex.to(device)
            LPIPS = torch.mean(loss_fn_alex(result, image)).item()

            LPIPS_SUM += LPIPS
            PSNR_SUM += PSNR
            SSIM_SUM += SSIM
            COUNT += 1
            print('Count:%s\tPSNR = %.4f\tLPIPS = %.4f\tSSIM = %.4f\t' % (COUNT, PSNR, LPIPS, SSIM,))

            # grid.append(result)
            # grid.append(encoded)
            # grid.append(image)

            image_show = tensor2im(image)
            encoded_show = tensor2im(encoded)
            mask_show = tensor2im(mask)
            se_show = tensor2im(rere)
            joint_show = tensor2im(joint)
            result_show = tensor2im(result)
            dec_show = tensor2im(dec)
            name = filename.split('\\')[-1] if filename.find('/') == -1 else filename.split('/')[-1]
            save_name = name.split('.')[0]

            save_path_image = args_this.save_dir + '/' + save_name + '.jpg'
            # save_path_mask = args_this.save_dir + '/' + save_name + '_mask.png'
            save_path_result = args_this.save_dir + '/' + save_name + '.jpg'
            save_path_dec = args_this.save_dir + '/' + save_name + '.jpg'
            # save_path_atten = args_this.save_dir + '/' + save_name + '_atten.jpg'

            # image_show = cv2.cvtColor(image_show, cv2.COLOR_RGB2BGR)
            # mask_show = cv2.cvtColor(mask_show, cv2.COLOR_RGB2BGR)
            # result_show = cv2.cvtColor(result_show, cv2.COLOR_RGB2BGR)

            # cv2.imwrite(save_path_image, image_show, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            # cv2.imwrite(save_path_mask, mask_show, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            # cv2.imwrite(save_path_result, result_show, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            # cv2.imwrite(save_path_dec, dec_show, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            # cv2.imwrite(save_path_atten, att_show, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            # save_result = transforms.ToPILImage()(image.squeeze(0))
            # save_result.save(save_path_image, quality=100)
            # save_mask = transforms.ToPILImage()(mask.squeeze(0))
            # save_mask.save(save_path_mask, quality=100)
            # save_result = transforms.ToPILImage()(result.squeeze(0))
            # save_result.save(save_path_result, quality=100)
            save_result = transforms.ToPILImage()(result.squeeze(0))
            save_result.save(save_path_dec, quality=100)

            if dir_path is None:
                plt.figure(1)
                plt.subplot(3, 4, 1)
                plt.axis('off')
                plt.imshow(image_show)
                plt.subplot(3, 4, 2)
                plt.axis('off')
                plt.imshow(encoded_show)
                plt.subplot(3, 4, 3)
                plt.axis('off')
                plt.imshow(mask_show)
                plt.subplot(3, 4, 4)
                plt.axis('off')
                plt.imshow(se_show)
                plt.subplot(3, 4, 5)
                plt.axis('off')
                plt.imshow(result_show)
                plt.subplot(3, 4, 6)
                plt.axis('off')
                # plt.imshow(t_img_show)
                # plt.subplot(3, 4, 7)
                # plt.axis('off')
                # plt.imshow(warped_img_show)
                # plt.subplot(3, 4, 8)
                # plt.axis('off')
                # plt.imshow(rectify_img_show)
                plt.subplot(3, 4, 9)
                plt.axis('off')
                plt.imshow(dec_show)
                plt.subplot(3, 4, 10)
                plt.axis('off')
                # plt.imshow(noised_img_show)
                # plt.subplot(3, 4, 11)
                # plt.axis('off')
                # plt.imshow(att_show)
                plt.show()

        # grid_show = torch.cat(grid, dim=0)
        # grid_save = torchvision.utils.make_grid(grid_show, nrow=1, padding=5)
        # path = save_path + '/psnr29.png'
        # torchvision.utils.save_image(grid_save, fp=path, nrow=1, padding=5)

        # result_partial = partialIMG.detach().cpu().numpy()
        # result_partial = np.transpose(result_partial, (0, 2, 3, 1))
        # result_partial = (np.clip(result_partial[0], 0.0, 1.0) * 255).astype(np.uint8)
        # result_partial = cv2.cvtColor(result_partial, cv2.COLOR_RGB2BGR)
        #
        # result_partialMask = partialMask.detach().cpu().numpy()
        # result_partialMask = np.transpose(result_partialMask, (0, 2, 3, 1))
        # result_partialMask = (np.clip(result_partialMask[0], 0.0, 1.0) * 255).astype(np.uint8)
        # result_partialMask = cv2.cvtColor(result_partialMask, cv2.COLOR_RGB2BGR)

        # 打印PSNR,SSIM等信息
        print('img_nums:%s\tPSNR = %.4f\tLPIPS = %.4f\tSSIM = %.4f\t' % (
            COUNT, PSNR_SUM / COUNT, LPIPS_SUM / COUNT, SSIM_SUM / COUNT,))


if __name__ == "__main__":
    img_size = 400
    device = torch.device("cuda:1")
    opt = options.getOpt()
    # cp_deep_path = '/opt/data/mingjin/pycharm/Net/output/Vanilla/model_final_end501_Warp0.2_Vanilla.pth'
    # cp_deep_path = r'D:\BaiduNetdiskWorkspace\PycharmProject\Net\exp\100.pth'
    # cp_deep_path = '/opt/data/mingjin/pycharm/Net/output/Vanilla/model_final_end501_Warp0.2AndAllNoise_version3_Vanilla.pth'
    # cp_path = r'D:\BaiduNetdiskWorkspace\PycharmProject\Net\exp\1139.pth'  # 方案2
    # cp_path = '/opt/data/mingjin/pycharm/Net/output/retrain the previous model_1/1255.pth'  # 方案1
    # cp_path = '/opt/data/mingjin/pycharm/MyFirstNet/output/6_DDHNewAdjust5_Parameter/326.pth'  # psnr 26.34
    # cp_path = '/opt/data/mingjin/pycharm/MyFirstNet/output/6_DDHNewAdjust5_Parameter/327.pth'  # psnr 27.14
    # cp_path = '/opt/data/mingjin/pycharm/MyFirstNet/output/6_DDHNewAdjust5_Parameter/328.pth'  # psnr 29.31
    # cp_path = '/opt/data/mingjin/pycharm/MyFirstNet/output/6_DDHNewAdjust5_Parameter/335.pth'  # psnr 30.3111
    cp_path = '/opt/data/mingjin/pycharm/MyFirstNet/output/6_DDHNewAdjust5_Parameter/1138.pth'  # psnr 33.3111
    # cp_path = '/opt/data/mingjin/pycharm/MyFirstNet/output/6_DDHNewAdjust5_Parameter/325.pth'  # psnr 25.7252
    # cp_path = '/opt/data/mingjin/pycharm/MyFirstNet/output/6_DDHNewAdjust5_Parameter/335.pth'  # psnr 25
    # cp_path = '/opt/data/mingjin/pycharm/MyFirstNet/output/6_DDHNewAdjust5_Parameter/585.pth'  # psnr 28
    # cp_path = '/opt/data/mingjin/pycharm/MyFirstNet/output/6_DDHNewAdjust5_Parameter/330.pth'  # 50bits
    # cp_path = '/opt/data/mingjin/pycharm/MyFirstNet/output/6_DDHNewAdjust5_Parameter/325.pth'  # 100bits
    # img_path = '/opt/data/mingjin/pycharm/Net/qizi.png'
    # img_path = '/opt/data/mingjin/pycharm/MyFirstNet/sample/origin/7.jpg'
    # img_path = '/opt/data/mingjin/pycharm/Data/HiDDeN/train/000000001319.jpg'
    # img_path = '/opt/data/mingjin/pycharm/Net/exp/sample/c4/reality/3.jpg'
    # img_path = r'D:\BaiduNetdiskWorkspace\PycharmProject\Net\exp\sample\origin\17shuangquyu.jpg'
    # cp_path = '/opt/data/mingjin/pycharm/MyFirstNet/output/6_DDHNewAdjust5_Parameter/1139.pth'  # 方案2
    # img_path = None
    # img_path = '/home/yizhi/Net/image_folder/Wild/origin/1.jpg'#1个
    # img_path = '/home/yizhi/Net/image_folder/Wild/origin/2.jpg'  # 1个
    # img_path = '/home/yizhi/Net/image_folder/Wild/origin/3.jpg'  # 1个
    # img_path = '/home/yizhi/Net/image_folder/Wild/origin/4.jpg'  # 1个
    # img_path = '/home/yizhi/Net/image_folder/Wild/origin/5.jpg'  # 1个
    # img_path = '/home/yizhi/Net/image_folder/Wild/origin/6.jpg'  # 1个
    # img_path = '/home/yizhi/Net/image_folder/Wild/origin/7.jpg'  # 1个
    # img_path = '/home/yizhi/Net/image_folder/Wild/origin/8.jpg'  # 1个
    # img_path = '/home/yizhi/Net/image_folder/Wild/origin/9.jpg'  # 1个
    # img_path = '/home/yizhi/Net/image_folder/Wild/origin/11.jpg'  # 1个
    # img_path = '/home/yizhi/Net/image_folder/Wild/origin/16.jpg'  # 2个
    # img_path = '/home/yizhi/Net/image_folder/Wild/origin/17shuangquyu.jpg'  # 2个
    # img_path = '/home/yizhi/Net/image_folder/Wild/origin/28.jpg'  # 2个
    # img_path = '/home/yizhi/Net/image_folder/Wild/origin/32.jpg'  # 2个
    # img_path = '/home/yizhi/Net/image_folder/Wild/origin/17.jpg'  # 2个(fa
    # img_path = '/home/yizhi/Net/image_folder/Wild/origin/25.jpg'  # 2个(fa
    # img_path = '/home/yizhi/Net/image_folder/Wild/origin/10.jpg'  # 2个(fa
    # img_path = '/home/yizhi/Net/image_folder/Wild/origin/e.jpg'  # 2个(fa
    # img_path = '/home/yizhi/Net/image_folder/Wild/origin/13.jpg'  # 2个(fa
    # img_path = '/home/yizhi/Net/image_folder/Wild/origin/12.jpg'  # 2个(fa


    img_path = '/home/yizhi/Net/image_folder/Wild/origin/12.jpg'
    # img_path = '/opt/data/mingjin/pycharm/Net/exp/sample/origin/1.jpg'
    # dir_path = '/opt/data/mingjin/pycharm/Net/exp/sample/origin/'
    # dir_path = '/opt/data/mingjin/pycharm/Net/exp/sample/visual quality/origion'
    # dir_path = '/home/yizhi/image_folder/Wild/origin'
    dir_path = None
    # dir_path = '/home/yizhi/bizhi'
    # save_path = '/opt/data/mingjin/pycharm/Net/exp/sample/visual quality/ours'
    # dir_path = '/opt/data/mingjin/pycharm/Data/HiDDeN/val'
    # dir_path = '/opt/data/mingjin/pycharm/Data/HiDDeN/test'
    # dir_path = '/opt/data/mingjin/pycharm/Net/exp/sample/c4/exp2-3'
    # save_path = '/opt/data/mingjin/pycharm/Net/image_folder/encoded'
    # save_path = '/home/yizhi/bizhi/new'
    # dir_path = None
    # dir_path = '/opt/data/mingjin/pycharm/Net/exp/sample/c4/reality'
    # dir_path = '/opt/data/mingjin/pycharm/Data/HiDDeN/test'
    # save_path = r'D:\BaiduNetdiskWorkspace\PycharmProject\Net\image_folder\encoded'
    save_path = '/home/yizhi/Net/duibi012shuiyin'
    main(cp_path, img_path, dir_path, save_path, img_size, 4)
