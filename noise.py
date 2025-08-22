import os
import random

import cv2
import numpy as np
import pywt
import torch
import torchvision
from matplotlib import pyplot as plt
from scipy import signal
from pytorch_wavelets import DWTForward, DWTInverse
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from utils import get_rand_homography_mat, get_min_max_scale_extract, get_cdos, tensor2im, Random_noise, Gauss_noise, im2tensor, get_color_distortion, \
    generate_random_key
import torchgeometry
import kornia as K
# from noise_layers.compose import Compose
from exp.noise_layers_exp.jpeg import JPEGCompress
# from args import options
# from noise_layers.resizeThenBack import RESIZE


def noise_layer(img, jpeg_folder='./jpeg_folder/', key=None):
    device = img.device
    batch = img.shape[0]
    # group = batch // 5
    # ori_img = img[0:group, ...].clone()
    # trans_img = img[group:batch, ...].clone()
    # noised_img = trans_img
    noised_img = img
    # print(ori_img.shape, noised_img.shape)
    # # - 噪声 - #
    # k_size_gauss = random.choice([3, 5, 7])
    # k_size_motion = random.choice([3, 5, 7])
    # angle = torch.Tensor(noised_img.size()[0]).uniform_(-35, 35)
    # direction = torch.from_numpy(np.random.choice([1, 0, -1], noised_img.size()[0])).float()
    #
    # noised_img = noised_img if random.random() < 0.2 else K.filters.gaussian_blur2d(noised_img, (k_size_gauss, k_size_gauss), (0.1, 1.0))
    # # print(noised_img.shape)
    # noised_img = noised_img if random.random() < 0.2 else K.filters.motion_blur(noised_img, k_size_motion, angle, direction)
    # #
    # # - 模糊 - #
    # random_noise_scale = torch.Tensor(1).uniform_(-0.1, 0.1).item()
    # gauss_noise_scale = torch.Tensor(1).uniform_(0., 0.01).item()
    #
    # # noised_img = noised_img if random.random() < 0.2 else Random_noise(noised_img, random_noise_scale)
    # noised_img = noised_img if random.random() < 0.5 else Gauss_noise(noised_img, gauss_noise_scale)
    #
    # # - 颜色变换 - #
    # # contrast = [-0.3, 0.3]
    # # color = [-0.1, 0.1]
    # # brightness = [-0.3, 0.3]
    # contrast = [-0.2, 0.3]
    # color = [-0.05, 0.1]
    # brightness = [-0.2, 0.3]
    # con_scale = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
    # col_scale = torch.tensor(1.0).uniform_(color[0], color[1]).item()
    # bri_scale = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
    #
    # noised_img = (1 + con_scale) * (noised_img + col_scale) + bri_scale
    # # noised_img = (1 - 0.2) * (noised_img - 0.05) - 0.2
    # # noised_img = get_min_max_scale_extract(noised_img)
    # noised_img = torch.clip(noised_img, 0.0, 1.0)

    # - jpeg压缩 - #
    qf = int(100. - torch.rand(1)[0] * 50)  # 50-100之间均匀采样的量化因子
    noised_img = JPEGCompress(noised_img, qf, device)
    # # UDH JPEG
    # jpeg_folder_path = jpeg_folder + '/' + key + '/'
    # if not os.path.exists(jpeg_folder_path):
    #     os.makedirs(jpeg_folder_path)
    #
    # container_img_copy = noised_img.clone()
    # containers_ori = container_img_copy.detach().cpu().numpy()
    # containers = np.transpose(containers_ori, (0, 2, 3, 1))
    # containers = (np.clip(containers, 0.0, 1.0) * 255).astype(np.uint8)
    # N = noised_img.shape[0]
    # for i in range(N):
    #     qf = int(100. - torch.rand(1)[0] * 50)  # 50-100之间均匀采样的量化因子
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

    # # - 透视扭曲 - #
    # warp = [0.0, 0.2]
    # warp_scale = torch.tensor(1.0).uniform_(warp[0], warp[1]).item()
    # h, w = img.shape[2:4]
    # homography = get_rand_homography_mat(img, warp_scale)
    # homography = torch.from_numpy(homography).float()
    # noised_img = torchgeometry.warp_perspective(img if random.random() < 0.3 else noised_img, homography[:, 1, :, :], dsize=(h, w))

    # noised_img = trans_img if random.random() < 0.2 else noised_img
    # return img if random.random() < 0.2 else torch.cat([ori_img, noised_img], dim=0).to(device)
    return noised_img


if __name__ == '__main__':
    # img_path = r'D:\BaiduNetdiskWorkspace\PycharmProject\Net\exp\sample\origin\19.jpg'
    img_path = '/home/yizhi/Net/stegastamp0/StegaStamp-pytorch/encoded_img.jpg'
    img = Image.open(img_path)
    transform = transforms.Compose([
        transforms.Resize((400, 400), Image.BICUBIC),
        transforms.ToTensor()])
    img = transform(img).unsqueeze(0)
    # img = torch.rand(10, 3, 400, 400)
    im = noise_layer(img)
    print(im.shape)
    if im.shape[0] == 1:
        im_np = tensor2im(im)

        plt.figure(1)
        plt.subplot(1, 1, 1)
        plt.axis('off')
        plt.imshow(im_np)
        plt.savefig('/home/yizhi/Net/stegastamp0/StegaStamp-pytorch/noise.jpg')
        plt.show()
