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
from points_embed import template_embed, template_extract#原来是注释的
from utils import get_cdos, get_warped_points_and_gap, get_rand_homography_mat, tensor2im, get_secret_acc, get_warped_extracted_IMG, im2tensor, Gauss_noise, \
    Random_noise
from args import options
from noise_layers.compose import Compose
from points_rgb import template_embed_rgb as template_embed, template_extract_rgb as template_extract, genPN#原来是注释的
from points_Ycbcr import template_embed_ycbcr as template_embed, template_extract_ycbcr as template_extract, genPN
from noise import noise_layer
from torchvision import transforms
from PIL import Image, ImageOps
import torchgeometry
import glob
import os
import cv2
import numpy as np
# from kornia.losses import psnr,ssim
from kornia import losses#原来的
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


def main(cp_path=None, cp_deep_path=None, img_path=None, dir_path=None, save_dir=None, img_size=256, p_nums=4):
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

            # if (torch.sum(mask == 1) / (3 * 400 * 400)) < 1 / 8:
            #     continue

            result = encoded * mask + image * (1 - mask)

            f = torch.from_numpy(np.ones((3, 3, 11, 11), dtype=np.float32)).to(device)
            mask = F.conv2d(mask, f, bias=None, padding=20, dilation=4)
            mask = F.interpolate(mask, (img_size, img_size))
            mask[mask > 0] = 1.0

            # f = torch.from_numpy(np.ones((3, 3, 7, 7), dtype=np.float32)).to(device)
            # mask = F.conv2d(mask, f, bias=None, padding=3, dilation=2)
            # mask = F.interpolate(mask, (img_size, img_size))
            # mask[mask > 0] = 1.0

            # f = torch.from_numpy(np.ones((3, 3, 4, 4), dtype=np.float32)).to(device)
            # noised_mask = F.conv2d(mask, f, bias=None, padding=3, dilation=2)
            # mask = F.interpolate(mask, (img_size, img_size))
            # noised_mask[noised_mask > 0] = 1.0

            # result = (255.0 * torch.clamp(result, 0.0, 1.0)).long()
            # result = result.float() / 255.0

            ###############################################################################
            # # deep model
            t_img = result
            deep_model = DeepHomographyModel(p_nums).to(device)
            checkpoint = torch.load(cp_deep_path, map_location=torch.device('cpu'))
            deep_model.load_state_dict(checkpoint['model'])
            deep_model.eval()
            deep_model.to(device)

            item_ones = torch.ones_like(result).float().to(device)
            # item_ones = torch.ones(1, 3, 400, 400).float().to(device)
            #

            ######### perspective warp ############
            h, w = result.shape[2:4]
            homography = get_rand_homography_mat(result, 0.1)#hhhhh
            homography = torch.from_numpy(homography).float().to(device)
            # # homography = get_rand_homography(result, 25, 25)
            # # homography = torch.from_numpy(homography).float().to(device)
            #
            source_points = get_cdos(result, p_nums)
            gap, warped_points = get_warped_points_and_gap(source_points, homography)

            pn = genPN(7)
            t_img, PN1 = template_embed(result, source_points, PN=pn)
            t_ones, PN2 = template_embed(item_ones, source_points, PN=pn)

            #####################################################
            # noised
            # compose = Compose(opt, None).to(device)
            # noised_img = compose(t_img, 100000)
            # ----------------------------------------------- #
            # kornia的噪声
            # wait_warp = t_img
            wait_warp = result
           ###### scaling #######
            # wait_warp = noise_layer(wait_warp, key='sample_deep').to(device)
            resize = RESIZE()
            # # resize_scale = torch.Tensor(1).uniform_(0.9, 1.3).item()
            resize_scale = 2 #hhhhhhh
            wait_warp = resize(wait_warp, resize_scale)

           ####### blur #########
            # k_size_gauss = random.choice([3, 5, 7])
            k_size_gauss = 5#hhhhh
            wait_warp = K.filters.gaussian_blur2d(wait_warp, (k_size_gauss, k_size_gauss), (0.1, 1.0))#原来的
            # wait_warp = kornia.filters.gaussian_blur2d(wait_warp, (k_size_gauss, k_size_gauss), (0.1, 1.0))
            # k_size_motion = random.choice([3, 5, 7])
            k_size_motion = 5#hhhhhhh
            angle = torch.Tensor(wait_warp.size()[0]).uniform_(-35, 35)
            direction = torch.from_numpy(np.random.choice([1, 0, -1], wait_warp.size()[0])).float()
            wait_warp = K.filters.motion_blur(wait_warp, k_size_motion, angle, direction)

            ######### color ###########
            # contrast = [-0.2, 0.3]
            # color = [-0.05, 0.1]
            # brightness = [-0.2, 0.3]
            contrast = [-0.1, 0.1]#hhhhh
            color = [-0.03, 0.03]#hhhhhh
            brightness = [-0.1, 0.1]#jhhhhh
            con_scale = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
            col_scale = torch.tensor(1.0).uniform_(color[0], color[1]).item()
            bri_scale = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
            wait_warp = (1 + con_scale) * (wait_warp + col_scale) + bri_scale
            wait_warp = torch.clip(wait_warp, 0.0, 1.0)

            ########## noise ###########
            random_noise_scale = torch.Tensor(1).uniform_(-0.03, 0.03).item()#hhhhh
            gauss_noise_scale = torch.Tensor(1).uniform_(0., 0.007).item()#hhhhhh

            wait_warp = Random_noise(wait_warp, random_noise_scale)
            wait_warp = Gauss_noise(wait_warp, gauss_noise_scale)
            wait_warp = torch.clip(wait_warp, 0.0, 1.0)

            ######### jpeg #########
            # qf = int(100. - torch.rand(1)[0] * 50)
            qf = 50#hhhhhh
            wait_warp = JPEGCompress(wait_warp, qf, device)
            wait_warp = Color_Manipulation(args, wait_warp, 1000000)
################################分割线#########################################
            index = torch.where(mask == 1)
            h_min, h_max = torch.min(index[2]), torch.max(index[2])
            w_min, w_max = torch.min(index[3]), torch.max(index[3])
            weight = w_max - w_min
            height = h_max + 25 - (h_min - 25)
            exp_w = img_size * 0.6
            exp_h = img_size * 0.6

            print(exp_h, exp_w)
            print('aa' + str((weight * height) / (img_size * img_size)))
            ones = torch.zeros_like(image)

            ones[:, :, h_min: h_max, w_min: w_max] = 1
            wait_warp = wait_warp * ones + torch.ones_like(ones) * (1 - ones)

            wait_warp = torchgeometry.warp_perspective(wait_warp, homography[:, 1, :, :], dsize=(h, w))
            warp_mask = torchgeometry.warp_perspective(mask, homography[:, 1, :, :], dsize=(h, w))
            warp_mask[warp_mask > 0] = 1.0

            grid = [result, wait_warp, torch.abs(result - wait_warp) * 10]
            grid_show = torch.cat(grid, dim=0)
            grid_save = torchvision.utils.make_grid(grid_show, nrow=1, padding=5)
            save_name = filename.split('\\')[-1].split('.')[0] if filename.find('\\') != -1 else filename.split('/')[-1].split('.')[0]
            path = save_dir + '/' + 'combined' + '.jpg'
            torchvision.utils.save_image(grid_save, fp=path, nrow=1, padding=5)
            # # #####################################################
            start_time = time.time()
            # wait_warp = F.interpolate(wait_warp, (400, 400))
            Ip1 = template_extract(wait_warp, PN1).to(device)
            Ip2 = template_extract(t_ones, PN2).to(device)

            inputs = torch.cat([Ip1, Ip2], dim=1)
            predict_gap = deep_model.forward(inputs)
            predict_points = source_points + predict_gap.view(-1, p_nums, 2).detach().cpu().numpy()

            criterion = nn.MSELoss().to(device)
            ACE = criterion(predict_gap, gap.view(-1, 2 * p_nums)).item()

            m, _ = cv2.findHomography(np.float32(source_points).reshape(-1, 2), np.float32(predict_points).reshape(-1, 2))
            m_inverse = np.linalg.inv(m)
            m_inverse = torch.from_numpy(m_inverse).unsqueeze(0).float().to(device)
            rectify_img = torchgeometry.warp_perspective(wait_warp, m_inverse, dsize=(h, w))
            rectify_mask = torchgeometry.warp_perspective(warp_mask, m_inverse, dsize=(h, w))
            rectify_mask[rectify_mask > 0] = 1

            elapsed_time = time.time() - start_time
            TIME_SUM += elapsed_time
            ###############################################################################

            wait_dec = wait_warp
            # wait_mask = rCnnNet(wait_dec)
            # if wait_mask is None:
            #     continue
            # wait_mask = F.conv2d(wait_mask, f, bias=None, padding=20, dilation=4)
            # wait_mask = F.interpolate(wait_mask, (img_size, img_size))
            # wait_mask[wait_mask > 0] = 1.0
            # wait_mask = mask
            # wait_dec = t_img
            # wait_dec = result
            # start_time = time.time()
            # imReg, H = alignImage(tensor2im(wait_warp), tensor2im(result))
            # rec_img = im2tensor(imReg).to(device)
            # elapsed_time = time.time() - start_time
            # TIME_SUM += elapsed_time
            # wait_dec = rec_img

            # wait_dec = rectify_img
            wait_mask = rCnnNet(wait_dec)
            if wait_mask is None:
                continue
            wait_mask = F.conv2d(wait_mask, f, bias=None, padding=20, dilation=4)
            wait_mask = F.interpolate(wait_mask, (img_size, img_size))
            wait_mask[wait_mask > 0] = 1.0

            # wait_mask = rectify_mask
            dec = get_warped_extracted_IMG(wait_dec.clone(), wait_mask.clone(), args=opt, isMul=False, isHomography=False, isResize=False)
            pred_bit, attention_map = decoder(dec)

            bit_acc = get_secret_acc(pred_bit, sec)

            # 计算PSNR,SSIM
            PSNR = losses.psnr_loss(t_img, image, 1.).item()
            SSIM = torch.mean(losses.ssim_loss(t_img, image, 11)).item()#原来的
            # SSIM = torch.mean(ssim(t_img, image, 11)).item()
            loss_fn_alex = lpips.LPIPS(net='alex', verbose=False)
            loss_fn_alex = loss_fn_alex.to(device)
            LPIPS = torch.mean(loss_fn_alex(t_img, image)).item()

            # if bit_acc < 0.9:
            #     continue
            bit_SUM += bit_acc
            LPIPS_SUM += LPIPS
            PSNR_SUM += PSNR
            SSIM_SUM += SSIM
            # ACE_SUM += ACE
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
                # att_show = tensor2im(att)
                t_img_show = tensor2im(t_img)
                # warped_img_show = tensor2im(warped_img)
                noised_img_show = tensor2im(wait_warp)
                rectify_img_show = tensor2im(wait_dec)
                dec_show = tensor2im(dec)
                Ip1_show = tensor2im(Ip1)
                Ip2_show = tensor2im(Ip2)
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
                plt.imshow(t_img_show)
                plt.subplot(2, 5, 5)
                plt.axis('off')
                plt.imshow(mask_show)
                plt.subplot(2, 5, 6)
                plt.axis('off')
                plt.imshow(Ip1_show, cmap=plt.cm.gray)
                plt.subplot(2, 5, 7)
                plt.axis('off')
                plt.imshow(Ip2_show, cmap=plt.cm.gray)
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
    cp_deep_path = '/home/yizhi/output/deep_model/NoNoise_2_4points_addEncoder/300.pth'
    cp_path = '/opt/data/mingjin/pycharm/MyFirstNet/output/6_DDHNewAdjust5_Parameter/1138.pth'
    img_path = '/home/yizhi/exp/sample/origin/23.jpg'
    # img_path = '/home/yizhi/Data/HiDDeN/train/000000026497.jpg'
    # img_path = None
    dir_path = None
    # dir_path = '/opt/data/mingjin/pycharm/Data/HiDDeN/test'
    # dir_path = '/opt/data/mingjin/pycharm/MyFirstNet/sample/origin'
    # dir_path = '/opt/data/mingjin/pycharm/Data/HiDDeN/val'
    # dir_path = '/opt/data/mingjin/pycharm/Net/image_folder/Wild/origin'
    # save_path = '/opt/data/mingjin/pycharm/Net/sample/result/stega_demo'
    # save_path = '/opt/data/mingjin/pycharm/Net/paper/c4/exp1'
    # save_path = '/opt/data/mingjin/pycharm/Net/image_folder/Wild/origin'
    save_path = '/home/yizhi/image_folder/Simulation'
    # save_path = None
    main(cp_path, cp_deep_path, img_path, dir_path, save_path, img_size, 4)
