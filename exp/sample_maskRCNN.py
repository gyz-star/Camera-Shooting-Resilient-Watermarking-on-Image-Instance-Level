import glob
import os

import numpy as np
import torchvision
from matplotlib import pyplot as plt
from torchvision import transforms
from PIL import Image
import torch
import torch.nn.functional as F
import sys

sys.path.append("..")
from model import MaskRCNN
from noise_layers.compose import Compose
from utils import tensor2im, extract_img_mask
from args import options

device = torch.device('cuda:3')


def main(img_path, img_size, dir_path, save_dir):
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    if img_path is not None:
        files_list = [img_path]
    elif dir_path is not None:
        files_list = glob.glob(dir_path + '/*')
    else:
        print('Missing input image')
        return

    rCnnNet = MaskRCNN()
    rCnnNet.eval()
    rCnnNet.to(device)
    opt = options.getOpt()
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), Image.BICUBIC),
        transforms.ToTensor(), ])

    grid = []
    for filename in files_list:
        image = transform(Image.open(filename).convert('RGB')).unsqueeze(0).to(device)

        # noised
        compose = Compose(opt, None).to(device)
        noised_img = compose(image, 100000)

        mask_gt = rCnnNet(image)

        mask_ex = mask_gt.clone()
        f = torch.from_numpy(np.ones((3, 3, 7, 7), dtype=np.float32)).to(device)
        mask_ex = F.conv2d(mask_ex, f, bias=None, padding=6, dilation=2)
        mask_ex[mask_ex > 0] = 1.0
        # print(mask_ex.shape)

        ex_image, ex_mask = extract_img_mask(noised_img, mask_ex)
        grid.append(noised_img)
        grid.append(mask_gt)
        grid.append(mask_ex)
        grid.append(ex_image)
        grid.append(ex_mask)

        if dir_path is None:
            image_show = tensor2im(image)
            noised_show = tensor2im(noised_img)
            mask_gt_show = tensor2im(mask_gt)
            mask_ex_show = tensor2im(mask_ex)
            plt.figure(1)
            # plt.subplot(1, 2, 1)
            # plt.axis('off')
            # plt.imshow(image_show)
            # plt.subplot(2, 2, 2)
            # plt.axis('off')
            # plt.imshow(noised_show)
            # plt.subplot(1, 2, 2)
            plt.axis('off')
            plt.imshow(mask_gt_show)
            # plt.subplot(2, 2, 4)
            # plt.axis('off')
            # plt.imshow(mask_ex_show)
            plt.show()
    # print(grid[2].shape)
    # grid_show = torch.cat(grid, dim=0)
    grid_save = torchvision.utils.make_grid(mask_gt, nrow=5, padding=5)
    path = './maskDetection.png'
    torchvision.utils.save_image(grid_save, fp=path, nrow=5, padding=5)


if __name__ == '__main__':
    # img_path = '/home/yizhi/exp/sample/origin/12.jpg'
    # img_path = '/home/yizhi/image_folder/Wild/origin/8.jpg'
    # img_path = '/home/yizhi/bizhi/bizhi.jpg'
    img_path = '/home/yizhi/Net/image_folder/Wild/origin/1.jpg'
    dir_path = None
    save_path = './'
    main(img_path, 256, dir_path, save_path)
