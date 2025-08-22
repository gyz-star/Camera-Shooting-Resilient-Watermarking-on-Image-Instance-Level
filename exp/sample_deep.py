import glob
import os
import random
import sys
import numpy as np
import torch
from torch import nn

from exp.noise_layers_exp.jpeg import JPEGCompress
from exp.sample_stega import get_rand_homography
from utils import Gauss_noise, Random_noise

sys.path.append("../")
from model import DeepHomographyModel
from dataset_deep_model import MyDataset
import cv2
import matplotlib.pyplot as plt
from points_Ycbcr import template_embed_ycbcr as template_embed, template_extract_ycbcr as template_extract, genPN
# from points_yuv import template_embed, template_extract, genPN
from utils import get_cdos, get_warped_points_and_gap, get_rand_homography_mat, tensor2im
from noise_layers_exp.compose import Compose
from args import options
from torchvision import transforms
from PIL import Image, ImageOps
import torchgeometry
from noise import noise_layer
import kornia as K
import torch.nn.functional as F
from noise_layers_exp.resizeThenBack import RESIZE


def main(cp_path, img_path, dir_path, img_size=400, device=torch.device('cpu'), p_nums=4):
    if img_path is not None:
        files_list = [img_path]
    elif dir_path is not None:
        files_list = glob.glob(dir_path + '/*')
    else:
        print('Missing input image')
        return

    net = DeepHomographyModel(p_nums).to(device)
    checkpoint = torch.load(cp_path, map_location=torch.device('cpu'))
    # pn = checkpoint['pn_embed']
    pn = genPN(7)
    net.load_state_dict(checkpoint['model'])
    net.eval()

    transform = transforms.Compose([transforms.Resize((img_size, img_size), Image.BICUBIC),
                                    transforms.ToTensor()])
    ACE = 0
    COUNT = 0
    for filename in files_list:
        item_img = transform(Image.open(filename).convert('RGB')).unsqueeze(0).to(device)
        item_ones = torch.ones_like(item_img).float().to(device)

        h, w = item_img.shape[2:4]
        homography = get_rand_homography_mat(item_img, 0.2)
        homography = torch.from_numpy(homography).float().to(device)

        source_points = get_cdos(item_img, p_nums)
        gap, warped_points = get_warped_points_and_gap(source_points, homography)

        t_img, PN1 = template_embed(item_img, source_points, PN=pn)
        t_ones, PN2 = template_embed(item_ones, source_points, PN=pn)

        noised_img = t_img

        ###############################################################
        # 模拟加噪
        noised_img = noise_layer(noised_img, key='sample_deep').to(device)
        # jpeg_folder_path = '../jpeg_folder/'
        #
        # k_size_gauss = 7
        # k_size_motion = 7
        # angle = 35
        # direction = 0
        #
        # noised_img = K.filters.gaussian_blur2d(noised_img, (k_size_gauss, k_size_gauss), (0.1, 1.0))
        # # print(noised_img.shape)
        # noised_img = K.filters.motion_blur(noised_img, k_size_motion, angle, direction)
        # #
        # # - 模糊 - #
        # random_noise_scale = 0.1
        # gauss_noise_scale = 0.01
        #
        # noised_img = Random_noise(noised_img, random_noise_scale)
        # noised_img = Gauss_noise(noised_img, gauss_noise_scale)
        #
        # # - 颜色变换 - #
        # # contrast = [-0.3, 0.3]
        # # color = [-0.1, 0.1]
        # # brightness = [-0.3, 0.3]
        # con_scale = 0.15
        # col_scale = 0.05
        # bri_scale = 0.15
        #
        # noised_img = (1 + con_scale) * (noised_img + col_scale) + bri_scale
        # # noised_img = get_min_max_scale_extract(noised_img)
        # noised_img = torch.clamp(noised_img, 0.0, 1.0)
        #
        # # - jpeg压缩 - #
        # if not os.path.exists(jpeg_folder_path):
        #     os.makedirs(jpeg_folder_path)
        #
        # container_img_copy = noised_img.clone()
        # containers_ori = container_img_copy.detach().cpu().numpy()
        # containers = np.transpose(containers_ori, (0, 2, 3, 1))
        # containers = (np.clip(containers, 0.0, 1.0) * 255).astype(np.uint8)
        # N = noised_img.shape[0]
        # for i in range(N):
        #     # qf = int(100. - torch.rand(1)[0] * 50)  # 50-100之间均匀采样的量化因子
        #     qf = 70
        #     img_ = cv2.cvtColor(containers[i], cv2.COLOR_RGB2BGR)
        #     folder_img = jpeg_folder_path + str(i) + ".jpg"
        #     cv2.imwrite(folder_img, img_, [int(cv2.IMWRITE_JPEG_QUALITY), qf])
        #
        # containers_loaded = np.copy(containers)
        #
        # for i in range(N):
        #     folder_img = jpeg_folder_path + str(i) + ".jpg"
        #     img_ = cv2.imread(folder_img)
        #     containers_loaded[i] = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
        #
        # containers_loaded = containers_loaded.astype(np.float32) / 255
        # # containers_loaded = containers_loaded * 2 - 1 # transform range of containers from [0, 1] to [-1, 1]
        # containers_loaded = np.transpose(containers_loaded, (0, 3, 1, 2))
        #
        # container_gap = containers_loaded - containers_ori
        # container_gap = torch.from_numpy(container_gap).float().to(device)
        #
        # noised_img = noised_img + container_gap
        # noised_img = torch.clamp(noised_img, 0.0, 1.0)

        # compose = Compose(opt, None).to(device)
        # wait_warp = compose(wait_warp, 100000)

        # homography = get_rand_homography(item_img, 15, 15)
        # homography = torch.from_numpy(homography).float().to(device)
        # resize = RESIZE()
        # wait_warp = resize(wait_warp, 1.5)

        # wait_warp = JPEGCompress(wait_warp, 90, device)

        warped_img = noised_img
        # warped_img = t_img
        warped_img = torchgeometry.warp_perspective(warped_img, homography[:, 1, :, :], dsize=(h, w))
        # warped_img_d = torchgeometry.warp_perspective(t_img, homography[:, 1, :, :], dsize=(h, w))
        ###############################################################

        # warped_img = torchgeometry.warp_perspective(wait_warp, homography[:, 1, :, :], dsize=(h, w))

        Ip1 = template_extract(warped_img, PN1).to(device)
        Ip2 = template_extract(t_ones, PN2).to(device)
        inputs = torch.cat([Ip1, Ip2], dim=1)

        predict_gap = net.forward(inputs)
        criterion = nn.MSELoss().to(device)
        loss = criterion(predict_gap, gap.view(-1, 2 * p_nums))
        ace = loss.item() ** 0.5
        ACE += ace
        print('ace:' + str(ace))

        predict_points = source_points + predict_gap.view(-1, p_nums, 2).detach().cpu().numpy()

        if dir_path is None:
            # 看一下三者的差距
            source_points_ = np.array(source_points, np.int32).reshape((1, p_nums, 2))  # 数据类型必须为 int32
            warped_points_ = np.array(warped_points[0], np.int32).reshape((1, p_nums, 2))  # 数据类型必须为 int32
            predict_points_ = np.array(predict_points[0], np.int32).reshape((1, p_nums, 2))

            print(source_points_)
            print(warped_points_)
            print(predict_points_)

            m, _ = cv2.findHomography(np.float32(source_points).reshape(-1, 2), np.float32(predict_points).reshape(-1, 2))
            m_inverse = np.linalg.inv(m)
            m_inverse = torch.from_numpy(m_inverse).float().to(device)
            rectify_img = torchgeometry.warp_perspective(warped_img, m_inverse, dsize=(h, w))

            ones = torch.ones_like(t_img).to(device)
            m = torch.from_numpy(m).float().to(device)
            warp_ones = torchgeometry.warp_perspective(ones, m, dsize=(h, w))
            warp_ones = torchgeometry.warp_perspective(warp_ones, m_inverse, dsize=(h, w))

            rectify_img = rectify_img + (1 - warp_ones) * item_img

            image_np = tensor2im(item_img.detach())
            timg_np = tensor2im(t_img.detach())
            warped_img_np = tensor2im(warped_img.detach())
            pos_img_np = tensor2im(Ip1.detach())
            pos_gt_np = tensor2im(Ip2.detach())
            rectify_img_np = tensor2im(rectify_img.detach())

            # warped_img_np = cv2.polylines(warped_img_np.copy(), warped_points_, True, (255, 0, 0), 1, cv2.LINE_AA)
            # warped_img_np = cv2.polylines(warped_img_np.copy(), predict_points_, True, (0, 255, 0), 1, cv2.LINE_AA)
            Ip1_show = torch.cat([Ip1, Ip1, Ip1], dim=1)
            Ip1_show = tensor2im(Ip1_show)
            # Ip1_show_np = cv2.polylines(Ip1_show.copy(), warped_points_, True, (255, 0, 0), 1, cv2.LINE_AA)
            # Ip1_show_np = cv2.polylines(Ip1_show_np.copy(), predict_points_, True, (0, 255, 0), 1, cv2.LINE_AA)

            plt.figure(1)
            # plt.subplot(2, 3, 1)
            ##plt.imshow(image_np)
            # plt.imshow(timg_np)
            # plt.subplot(2, 3, 2)
            # plt.imshow(warped_img_np)
            # plt.subplot(2, 3, 3)
            # plt.imshow(timg_np - image_np)  # warped_img_np
            # plt.subplot(2, 3, 4)
            # plt.imshow(pos_img_np, cmap=plt.cm.gray)
            ## plt.imshow(Ip1_show_np)
            # plt.subplot(2, 3, 5)
            # plt.imshow(pos_gt_np, cmap=plt.cm.gray)
            # plt.subplot(2, 3, 6)
            plt.imshow(rectify_img_np)
            plt.axis('off')
            plt.savefig('/home/yizhi/Net/bizhi/24new.jpg')
            plt.show()
        COUNT += 1
    cornerAvgError = ACE / COUNT
    print('COUNT:{:d},Corner Avg Error (CAE): {:.3f}'.format(COUNT, cornerAvgError))


# ##############################
if __name__ == '__main__':
    opt = options.getOpt()
    # cp_path = '/opt/data/mingjin/pycharm/Net/output/Vanilla/model_final_end501_Warp0.2_Vanilla.pth'
    # cp_path = '/opt/data/mingjin/pycharm/Net/output/deep_model/AllNoise_2_4points/50.pth'
    # cp_path = '/opt/data/mingjin/pycharm/Net/output/deep_model/AllNoise_1/500.pth'
    cp_path = '/home/yizhi/Net/output/deep_model/NoNoise_2_4points_addEncoder/200.pth'
    # cp_path = '/opt/data/mingjin/pycharm/Net/output/deep_model/AllNoise_2_4points/500.pth'
    # img_path = None
    # dir_path = '/opt/data/mingjin/pycharm/Data/HiDDeN/test'
    # img_path = '/home/yizhi/exp/sample/origin/24.jpg'
    img_path = '/home/yizhi/Net/bizhi/24.jpg'
    dir_path = None
    # cp_path = '/opt/data/mingjin/pycharm/Net/output/Vanilla/model_final_end501_Warp0.2AndAllNoise_version3_Vanilla.pth'
    # cp_path = '/opt/data/mingjin/pycharm/Net/output/Vanilla/model_final_end501_Warp0.2AndAllNoise_beginVersion3Continue500_version3Vanilla.pth'
    # cp_path = '/workshop/mingjin/pycharm/Net/output/deep_model/AllNoise_1/500.pth'
    # dir_path = '/workshop/mingjin/pycharm/Net/exp/sample/origin'

    # cp_path = r'D:\BaiduNetdiskWorkspace\PycharmProject\Net\exp\model_final_end501_Warp0.2_Vanilla.pth'

    # cp_path = r'D:\BaiduNetdiskWorkspace\PycharmProject\Net\output\deep_model\AllNoise_1\500.pth'
    # img_path = r'D:\BaiduNetdiskWorkspace\PycharmProject\Net\exp\sample\origin\17shuangquyu.jpg'
    # dir_path = r'D:\BaiduNetdiskWorkspace\PycharmProject\Net\exp\sample\origin\'
    # dir_path = None

    # img_path = '/workshop/mingjin/pycharm/Data/Coco_mini/val/000000001296.jpg'
    # dir_path = None
    # dir_path = '/opt/data/mingjin/pycharm/Data/HiDDeN/test'
    # dir_path = r"D:\BaiduNetdiskWorkspace\PycharmProject\Net\exp\sample\origin"
    device = torch.device('cuda:0')
    main(cp_path, img_path, dir_path, device=device, p_nums=4)
