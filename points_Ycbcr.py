import glob
import random

import cv2
import kornia
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
    Random_noise, RgbYcbcr, rgb2ycbcr_tensor, ycbcr2rgb_tensor, rgb_to_ycbcr_jpeg, ycbcr_to_rgb_jpeg, convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr, \
    get_cdos_according_mask
import torchgeometry
import kornia as K
# from noise_layers.compose import Compose
from exp.noise_layers_exp.compose import Compose
from exp.noise_layers_exp.jpeg import JPEGCompress
from args import options
from noise_layers.resizeThenBack import RESIZE
from noise import noise_layer
from points_rgb import template_embed_rgb as template_embed_r, template_extract_rgb as template_extract_r
from model import MaskRCNN
from points_rgb import template_embed_rgb, template_extract_rgb


# from noise_layers.resize import RESIZE


def genPN(sz):
    rd = np.random.RandomState(1)
    PN = rd.randint(0, 2, [sz, sz]).astype('double') * 2 - 1
    return PN  # size:[sz,sz]


def template_embed_ycbcr(img, pots, PN=genPN(16)):
    scale = 4  # Magnification of correlation

    # shape value of image should be even
    b = img.shape[0]
    if img.shape[-2] % 2 == 1:
        img = torch.cat([img[:, :, 1, :].contiguous().view([b, img.shape[1], 1, img.shape[-1]]), img], dim=-2)

    if img.shape[-1] % 2 == 1:
        img = torch.cat([img[:, :, :, 1].contiguous().view([b, img.shape[1], img.shape[-2], 1]), img], dim=-1)

    PNwid = PN.shape[0]
    device = img.device

    dwt = DWTForward(J=1, wave='db2', mode='symmetric').to(device)

    img_YUV_tensor = convert_rgb_to_ycbcr(img)
    # img_YUV_tensor = kornia.color.rgb_to_ycbcr(img)
    img_dwt = img_YUV_tensor[:, 1, :, :].clone().unsqueeze(1)

    # img_YUV_tensor = rgb2ycbcr_tensor(img)
    # img_dwt = img_YUV_tensor[:, 2, :, :].clone().unsqueeze(1)

    yl, yh = dwt(img_dwt)  # LL, (LH, HL, HH)
    cov = yl

    PN_ = torch.from_numpy(PN).unsqueeze(0).unsqueeze(0).float().to(device)  # torch.Size([1, 1, 7, 7])
    R = F.conv2d(torch.pow(cov, 2), PN_, bias=None, padding=int((PNwid - 1) / 2))  # [1,1,9,9]

    t = torch.max(R) * scale + 5  # threshold

    z1 = torch.zeros(pots.shape[0])
    for i in range(pots.shape[0]):
        vetx = cov[:, :, pots[i, 0]:pots[i, 0] + PNwid, pots[i, 1]:pots[i, 1] + PNwid]
        z = torch.sum(torch.pow(vetx, 2) * PN_)  # z=x^T*W*x
        a = torch.sum(torch.pow(vetx, 2))  # a=x^T*x
        if z >= 0:
            z = 1e-5 if z == 0 else z
            alpha = (torch.sqrt(a * a - z * z + z * t) - a) / z
            vetx = vetx * (1.0 + alpha * PN_)
        else:
            alpha = (-torch.sqrt(a * a - z * z - z * t) + a) / z
            vetx = vetx * (1.0 - alpha * PN_)
        cov[:, :, pots[i, 0]:pots[i, 0] + PNwid, pots[i, 1]:pots[i, 1] + PNwid] = vetx
        z1[i] = torch.sum(torch.pow(vetx, 2) * PN_)

    idwt = DWTInverse(wave='db2', mode='symmetric').to(device)
    rec = idwt((yl, yh))[:, 0, ...].unsqueeze(1)

    wdimg = torch.cat([img_YUV_tensor[:, 0, :, :].unsqueeze(1), rec, img_YUV_tensor[:, 2, :, :].unsqueeze(1), ], dim=1)
    wdimg = convert_ycbcr_to_rgb(wdimg)
    # wdimg = kornia.color.ycbcr_to_rgb(wdimg)

    # wdimg = torch.cat([img_YUV_tensor[:, 0, :, :].unsqueeze(1), img_YUV_tensor[:, 1, :, :].unsqueeze(1), rec, ], dim=1)
    # wdimg = ycbcr2rgb_tensor(wdimg)

    wdimg = torch.clamp(wdimg, 0.0, 1.0)
    return wdimg, PN


def template_extract_ycbcr(img, PN):
    dimg = img.float()
    PNwid = PN.shape[0]
    device = img.device

    # img_YUV_tensor = rgb2ycbcr_tensor(dimg)

    img_YUV_tensor = convert_rgb_to_ycbcr(dimg)
    # img_YUV_tensor = kornia.color.rgb_to_ycbcr(dimg)

    dwt = DWTForward(J=1, wave='db2', mode='symmetric').to(device)
    # img_dwt = img_yuv[:, 1, :, :].clone().unsqueeze(1)
    img_dwt = img_YUV_tensor[:, 1, :, :].clone().unsqueeze(1)
    yl, yh = dwt(img_dwt)
    cov = yl
    PN_ = torch.from_numpy(PN).unsqueeze(0).unsqueeze(0).float().to(device)
    rst = F.conv2d(torch.pow(cov, 2), PN_, bias=None, padding=int((PNwid - 1) / 2))  # [1,1,9,9]
    # return get_min_max_scale_extract(rst)
    return rst


def demo(img_path, dir_path, img_size=400):
    if img_path is not None:
        files_list = [img_path]
    elif dir_path is not None:
        files_list = glob.glob(dir_path + '/*')
    else:
        print('Missing input image')
        return
    opt = options.getOpt()
    device = torch.device("cpu")
    count = 0
    PSNR = 0
    for filename in files_list:
        # imgpath = r'D:\BaiduNetdiskWorkspace\PycharmProject\Net\exp\sample\origin\12.jpg'
        # imgpath = '/workshop/mingjin/pycharm/Net/exp/sample/origin/1.jpg'
        # image = cv2.imread(imgpath)
        # image = cv2.resize(image, (400, 400))  # w*h
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # # image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        # img = torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2).float() / 255
        count += 1
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()])
        img = transform(Image.open(filename)).unsqueeze(0).to(device)

        # img = torch.ones(1, 3, 400, 400).float()
        img_ones = torch.ones(1, 3, img_size, img_size).float().to(device)
        h, w = img.shape[2:4]
        cdos = get_cdos(img)

        # MS = MaskRCNN().eval()
        # mask = MS(img)
        # cdos = np.array(get_cdos_according_mask(img, mask)).reshape(-1, 2)
        # cdos = np.array([[100, 100], [100, 150], [150, 100], [150, 150]])
        print(cdos)

        # PN = np.array([[1., 1., -1., -1., 1., 1., 1.],
        #                [1., 1., -1., -1., 1., -1., 1.],
        #                [1., -1., -1., 1., -1., -1., -1.],
        #                [1., -1., -1., 1., -1., -1., -1.],
        #                [1., -1., -1., -1., 1., 1., 1.],
        #                [1., 1., -1., -1., -1., 1., 1.],
        #                [1., 1., 1., 1., -1., 1., 1.]])
        PN = genPN(7)
        timg, PN = template_embed_ycbcr(img, cdos, PN)
        timg_ones, PN = template_embed_ycbcr(img_ones, cdos, PN)
        # timg_ones, PN = template_embed_rgb(img_ones, cdos, PN)
        # timg, PN = template_embed_r(img, cdos, PN)
        # timg_ones, PN = template_embed_r(img_ones, cdos, PN)
        psnr = K.losses.psnr(timg, img, 1.0)
        # psnr = calc_psnr(timg, img)
        print('psnr:' + str(psnr))
        PSNR += psnr
        # timg1 = F.interpolate(timg.float(), size=(200, 200))

        noised_img = timg
        # noised_img = noise_layer(noised_img, key='sample_points_ycbcr')

        # compose = Compose(opt, None)
        # noised_img = compose(noised_img, 100000)
        # resize = RESIZE()
        # noised_img = resize(noised_img, 4)
        # noised_img = F.interpolate(noised_img, size=(400, 400), mode='bilinear', align_corners=True)
        # noised_img = F.interpolate(noised_img, size=(256, 256), mode='bilinear', align_corners=True)
        # noised_img = K.enhance.adjust_brightness(noised_img, 0.2)
        # noised_img = K.enhance.adjust_hue(noised_img, 0.1)
        # noised_img = K.enhance.adjust_saturation(noised_img, 0.8)
        # noised_img = K.enhance.adjust_contrast(noised_img, 1.5)
        # color_distortion = get_color_distortion(s=0.01)
        # color_distortion = transforms.ColorJitter(0.3, 0.5, 0.1, 0)
        # noised_img = color_distortion(noised_img)
        # noised_img = K.filters.gaussian_blur2d(noised_img, (7, 7), (0.1, 1.0))
        # noised_img = K.filters.motion_blur(noised_img, 7, -30., 0)
        # noised_img = Random_noise(noised_img, 0.01)
        # noised_img = Gauss_noise(noised_img, 0.02)
        # noised_img = Random_noise(noised_img, 0.1)
        # noised_img = JPEGCompress(noised_img, 50, device)

        # contrast = [-0.3, 0.3]
        # color = [-0.1, 0.1]
        # brightness = [-0.3, 0.3]
        # con_scale = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
        # col_scale = torch.tensor(1.0).uniform_(color[0], color[1]).item()
        # bri_scale = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
        # noised_img = (1 + con_scale) * (noised_img + col_scale) + bri_scale
        # noised_img = (1 + 0.3) * (noised_img - 0.1) - 0.3
        # noised_img = torch.clamp(noised_img, 0.0, 1.0)
        # noised_img = get_min_max_scale_extract(noised_img)

        # noised_img = tensor2im(noised_img)
        # img_ = cv2.cvtColor(noised_img, cv2.COLOR_RGB2BGR)
        # folder_imgs = "1" + ".jpg"
        # cv2.imwrite(folder_imgs, img_, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        # img_ = cv2.imread(folder_imgs)
        # noised_img = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
        # noised_img = im2tensor(noised_img)
        #
        # hue_scale = torch.Tensor(noised_img.size()[0]).uniform_(-0.1, 0.1)

        #
        # resize = RESIZE()
        # resize_scale = torch.Tensor(1).uniform_(0.5, 1.5)
        # noised_img = resize(noised_img, resize_scale)
        # brightness_scale = torch.Tensor(noised_img.size()[0]).uniform_(-0.2, 0.2)
        # noised_img = K.enhance.adjust_brightness(noised_img, brightness_scale)
        # noised_img = K.enhance.adjust_hue(noised_img, hue_scale)
        # saturation_scale = torch.Tensor(noised_img.size()[0]).uniform_(0.8, 1.0)
        # noised_img = K.enhance.adjust_saturation(noised_img, saturation_scale)
        # contrast_scale = torch.Tensor(noised_img.size()[0]).uniform_(0.5, 1.5)
        # noised_img = K.enhance.adjust_contrast(noised_img, contrast_scale)
        # k_size_gauss = random.choice([3, 5, 7])
        # k_size_motion = random.choice([3, 5, 7])
        # angle = torch.Tensor(noised_img.size()[0]).uniform_(-30, 30)
        # direction = torch.from_numpy(np.random.choice([1, 0, -1], noised_img.size()[0])).float()
        # gauss_noise_scale = torch.Tensor(1).uniform_(0., 0.01).item()
        # noised_img = K.filters.gaussian_blur2d(noised_img, (k_size_gauss, k_size_gauss), (0.1, 1.0))
        # noised_img = K.filters.motion_blur(noised_img, k_size_motion, angle, direction)
        # noised_img = Gauss_noise(noised_img, gauss_noise_scale)

        # 透视扭曲
        wait_warp = noised_img
        # wait_warp = timg
        homography = get_rand_homography_mat(wait_warp, 0.1)
        homography = torch.from_numpy(homography).float().to(device)
        transformDec = torchgeometry.warp_perspective(wait_warp, homography[:, 1, :, :], dsize=(h, w))
        wait_extract = transformDec
        # wait_extract = timg

        # print(PN)
        PN = genPN(7)
        pots = template_extract_ycbcr(wait_extract, PN)
        # pots = template_extract_r(wait_extract, PN)
        # pots_gt = template_extract_ycbcr(timg, PN)
        pots_gt = template_extract_ycbcr(timg_ones, PN)
        # pots_gt = template_extract_rgb(timg_ones, PN)
        # print(pots.shape)

        if count == 1:
            pots_show = torch.cat([pots, pots, pots], dim=1)
            # print(pots_show.shape)
            pots_show = F.interpolate(pots_show, size=(img_size, img_size), mode='bilinear', align_corners=True)
            grid = [img, timg, (img - timg) * 5, pots_show]
            grid_show = torch.cat(grid, dim=0)
            grid_save = torchvision.utils.make_grid(grid_show, nrow=4, padding=5)
            name = filename.split('\\')[-1].split('.')[0] if filename.find('\\') != -1 else filename.split('/')[-1].split('.')[0]
            path = 'embeded_%s.png' % name
            torchvision.utils.save_image(grid_save, fp=path, nrow=4, padding=5)

        if img_path is not None:
            img_show = tensor2im(img)
            timg1_show = tensor2im(timg)
            gap_show = tensor2im(torch.abs(timg - img))
            noised_show = tensor2im(noised_img)
            pots_show = tensor2im(pots)
            pots_gt_show = tensor2im(pots_gt)
            warped_show = tensor2im(transformDec)

            for cdo in cdos:
                # print((cdo[0], cdo[1]))
                # print(img_show.dtype)
                img_show = cv2.circle(np.ascontiguousarray(img_show), (cdo[0], cdo[1]), 2, (0, 0, 255), -1)

            # warped_points_ = np.array(cdos, np.int32).reshape((1, 9, 2))
            # warped_img_np = cv2.polylines(timg1_show.copy(), warped_points_, True, (255, 0, 0), 1, cv2.LINE_AA)

            plt.figure(1)
            plt.subplot(2, 3, 1)
            # plt.axis('off')
            plt.imshow(img_show)
            plt.subplot(2, 3, 2)
            # plt.axis('off')
            plt.imshow(timg1_show)
            plt.subplot(2, 3, 3)
            # plt.axis('off')
            plt.imshow(timg1_show - img_show)
            plt.subplot(2, 3, 4)
            # plt.axis('off')
            plt.imshow(warped_show)
            plt.subplot(2, 3, 5)
            # plt.axis('off')
            plt.imshow(pots_show, cmap=plt.cm.gray)
            plt.subplot(2, 3, 6)
            # plt.axis('off')
            plt.imshow(pots_gt_show, cmap=plt.cm.gray)
            plt.show()

            timg = F.interpolate(timg, size=(400, 400), mode='bilinear', align_corners=True)
            noised_img = tensor2im(timg)
            name = filename.split('\\')[-1] if filename.find('/') == -1 else filename.split('/')[-1]
            save_name = name.split('.')[0]
            img_ = cv2.cvtColor(noised_img, cv2.COLOR_RGB2BGR)
            folder_imgs = "embed_%s_yuv_cb" % save_name + ".jpg"
            # cv2.imwrite(folder_imgs, img_, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    print('psnr_ave:' + str(PSNR / count))


if __name__ == '__main__':
    # img_path = None
    # dir_path = '/workshop/mingjin/pycharm/Net/exp/sample/origin/'
    # img_path = r'D:\BaiduNetdiskWorkspace\PycharmProject\Net\exp\sample\origin\2.jpg'
    img_path = '/opt/data/mingjin/pycharm/Net/exp/sample/origin/11.jpg'
    dir_path = None
    img_size = 400
    demo(img_path, dir_path, img_size)
