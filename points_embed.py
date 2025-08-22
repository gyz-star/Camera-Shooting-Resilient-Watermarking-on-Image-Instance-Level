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
    Random_noise
import torchgeometry
import kornia as K
# from noise_layers.compose import Compose
from exp.noise_layers_exp.compose import Compose
from exp.noise_layers_exp.jpeg import JPEGCompress
from args import options
from noise_layers.resizeThenBack import RESIZE
from noise import noise_layer


# from noise_layers.resize import RESIZE


def genPN(sz):
    rd = np.random.RandomState(1)
    PN = rd.randint(0, 2, [sz, sz]).astype('double') * 2 - 1
    return PN  # size:[sz,sz]


def template_embed(img, pots, PN=genPN(16), isOnes=False):
    scale = 4  # Magnification of correlation

    # shape value of image should be even
    b = img.shape[0]
    # if np.mod(img.shape[0], 2) == 1: # h变为偶数
    if img.shape[-2] % 2 == 1:
        # h*w*c
        # img = np.concatenate((np.reshape(img[1, :, :], [1, img.shape[1], img.shape[2]]), img), axis=0)
        # b*c*h*w
        img = torch.cat([img[:, :, 1, :].contiguous().view([b, img.shape[1], 1, img.shape[-1]]), img], dim=-2)

    # if np.mod(img.shape[1], 2) == 1:  # w变为偶数
    if img.shape[-1] % 2 == 1:
        # img = np.concatenate((np.reshape(img[:, 1, :], [img.shape[0], 1, img.shape[2]]), img), axis=1)
        img = torch.cat([img[:, :, :, 1].contiguous().view([b, img.shape[1], img.shape[-2], 1]), img], dim=-1)

    PNwid = PN.shape[0]
    # cA, (cH, cV, cD) = pywt.dwt2(img[:, :, 2], 'db2')
    # cov = cD
    device = img.device
    dwt = DWTForward(J=1, wave='db2', mode='symmetric').to(device)
    img_dwt = img[:, 2, :, :].clone().unsqueeze(1)
    # img_dwt = torch.cat([img_dwt, img_dwt, img_dwt], dim=1)  # 去掉
    # print(img_dwt.shape)
    yl, yh = dwt(img_dwt)  # LL, (LH, HL, HH)
    # print(yl.shape, yh[0].shape) # torch.Size([1, 1, 129, 129]) torch.Size([1, 1, 3, 129, 129])
    # cov = yh[0][:, :, 2, ...]  # yh的第2维
    cov = yl
    # print(cov.shape)

    # R = signal.convolve2d(pow(cov, 2), PN, mode='same', boundary='symm')
    PN_ = torch.from_numpy(PN).unsqueeze(0).unsqueeze(0).float().to(device)  # torch.Size([1, 1, 7, 7])
    # PN_ = torch.cat([PN_, PN_, PN_], dim=1)  # 去掉
    # print(PN_.shape)  #
    R = F.conv2d(torch.pow(cov, 2), PN_, bias=None, padding=int((PNwid - 1) / 2))  # [1,1,9,9]

    t = R.max() * scale + 5  # threshold
    # print('t:' + str(t))

    # z1 = np.zeros(pots.shape[0])
    z1 = torch.zeros(pots.shape[0])
    for i in range(pots.shape[0]):
        # vetx = cov[pots[i, 0]:pots[i, 0] + PNwid, pots[i, 1]:pots[i, 1] + PNwid]
        # print(pots[i])
        vetx = cov[:, :, pots[i, 0]:pots[i, 0] + PNwid, pots[i, 1]:pots[i, 1] + PNwid]
        # z = np.sum(pow(vetx, 2) * PN)  # z=x^T*W*x
        z = torch.sum(torch.pow(vetx, 2) * PN_)  # z=x^T*W*x
        # a = np.sum(pow(vetx, 2))  # a=x^T*x
        a = torch.sum(torch.pow(vetx, 2))  # a=x^T*x
        if z >= 0:
            # (1+alpha^2)*z + 2*alpha*a > threshold
            z = 1e-5 if z == 0 else z
            alpha = (torch.sqrt(a * a - z * z + z * t) - a) / z
            # alpha = np.clip(alpha, 0.2, 0.3)
            # alpha = np.clip(alpha, 30, 150)
            # alpha = 30 if alpha < 30 else alpha
            # print(alpha, z)
            # alpha = 200 if not isOnes else (np.sqrt(a * a - z * z + z * t) - a) / z
            # alpha = 150 if not isOnes else (np.sqrt(a * a - z * z + z * t) - a) / z  # 鲁棒性较强
            # alpha = 100.0 if not isOnes else (torch.sqrt(a * a - z * z + z * t) - a) / z
            # alpha = 25.0 if not isOnes else (torch.sqrt(a * a - z * z + z * t) - a) / z
            # print(alpha)
            vetx = vetx * (1.0 + alpha * PN_)
        else:
            # (1+alpha^2)*z - 2*alpha*a < -threshold
            alpha = (-torch.sqrt(a * a - z * z - z * t) + a) / z
            # alpha = np.clip(alpha, -0.3, -0.2)
            # alpha = -30 if alpha > -30 else alpha
            # alpha = np.clip(alpha, -150, -30)
            # print(alpha, z)
            # alpha = 200 if not isOnes else (np.sqrt(a * a - z * z + z * t) - a) / z
            # alpha = -150 if not isOnes else (np.sqrt(a * a - z * z + z * t) - a) / z
            # alpha = -100.0 if not isOnes else (-torch.sqrt(a * a - z * z + z * t) - a) / z
            # alpha = -25.0 if not isOnes else (-torch.sqrt(a * a - z * z + z * t) - a) / z
            # print(alpha)
            vetx = vetx * (1.0 - alpha * PN_)
        # cov[pots[i, 0]:pots[i, 0] + PNwid, pots[i, 1]:pots[i, 1] + PNwid] = vetx
        cov[:, :, pots[i, 0]:pots[i, 0] + PNwid, pots[i, 1]:pots[i, 1] + PNwid] = vetx
        # z1[i] = np.sum(pow(vetx, 2) * PN)
        z1[i] = torch.sum(torch.pow(vetx, 2) * PN_)

    # wdimg = np.array([img[:, :, 0], img[:, :, 1], pywt.idwt2((cA, (cH, cV, cD)), 'db2')])
    # wdimg = wdimg.transpose(1, 2, 0).astype('uint8') # torch的话不用进行这步操作了
    idwt = DWTInverse(wave='db2', mode='symmetric').to(device)
    rec = idwt((yl, yh))[:, 0, ...].unsqueeze(1)
    # print(idwt((yl, yh)).shape)
    # print(rec.shape) # torch.Size([1, 1, 256, 256])
    wdimg = torch.cat([img[:, 0, :, :].unsqueeze(1), img[:, 1, :, :].unsqueeze(1), rec], dim=1)
    # wdimg = torch.tensor(wdimg.clone(), dtype=torch.uint8)
    # wdimg = torch.clamp(wdimg, 0, 255).long()
    # wdimg = torch.clamp(wdimg, 0.0, 255.0)
    wdimg = torch.clamp(wdimg, 0.0, 1.0)
    return wdimg, PN


def template_extract(img, PN):
    dimg = img.float()
    PNwid = PN.shape[0]
    # cA, (cH, cV, cD) = pywt.dwt2(img[:, :, 2], 'db2')
    # cov = cD
    # rst = signal.convolve2d(pow(cov, 2), PN, mode='same', boundary='symm')
    device = img.device
    dwt = DWTForward(J=1, wave='db2', mode='symmetric').to(device)
    img_dwt = dimg[:, 2, :, :].clone().unsqueeze(1)
    # print(img_dwt.dtype)
    yl, yh = dwt(img_dwt)
    # cov = yh[0][:, :, 2, ...]
    cov = yl
    PN_ = torch.from_numpy(PN).unsqueeze(0).unsqueeze(0).float().to(device)
    rst = F.conv2d(torch.pow(cov, 2), PN_, bias=None, padding=int((PNwid - 1) / 2))  # [1,1,9,9]
    return get_min_max_scale_extract(rst)
    # return torch.sigmoid(rst)


if __name__ == '__main__':
    opt = options.getOpt()
    device = torch.device("cpu")
    imgpath = r'D:\BaiduNetdiskWorkspace\PycharmProject\Net\exp\sample\origin\22.jpg'
    # image = cv2.imread(imgpath)
    # image = cv2.resize(image, (256, 256))  # w*h
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # print(image.shape)

    img = Image.open(imgpath)
    transform = transforms.Compose([
        transforms.Resize((400, 400), Image.BICUBIC),
        transforms.ToTensor()])
    img = transform(img).unsqueeze(0)
    # print(img.shape)

    # img = torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2).float() / 255
    # img = torch.ones(1, 3, 256, 256)
    # img = torch.ones(1, 3, 200, 200).float().cuda() * 255
    # img.requires_grad = True
    # img1 = F.interpolate(img, size=(200, 200))
    h, w = img.shape[2:4]
    cdos = get_cdos(img)
    # cdos = np.array([[100, 100], [100, 150], [150, 100], [150, 150]])
    # print(cdos)

    # PN = np.array([[1., 1., -1., -1., 1., 1., 1.],
    #                [1., 1., -1., -1., 1., -1., 1.],
    #                [1., -1., -1., 1., -1., -1., -1.],
    #                [1., -1., -1., 1., -1., -1., -1.],
    #                [1., -1., -1., -1., 1., 1., 1.],
    #                [1., 1., -1., -1., -1., 1., 1.],
    #                [1., 1., 1., 1., -1., 1., 1.]])
    PN = genPN(7)
    timg, PN = template_embed(img, cdos, PN, isOnes=False)
    psnr = K.losses.psnr(timg, img, 1.0)
    print('psnr:' + str(psnr))
    # timg1 = F.interpolate(timg.float(), size=(200, 200))

    noised_img = timg
    # noised_img = noise_layer(noised_img)

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
    homography = get_rand_homography_mat(wait_warp, 0.2)
    homography = torch.from_numpy(homography).float()
    transformDec = torchgeometry.warp_perspective(wait_warp, homography[:, 1, :, :], dsize=(h, w))

    # wait_extract = noised_img
    wait_extract = transformDec
    # wait_extract = timg

    # print(PN)
    pots = template_extract(wait_extract, PN)
    pots_gt = template_extract(timg, PN)
    print(pots.shape)

    # pots_show = torch.cat([pots, pots, pots], dim=1)
    # print(pots_show.shape)
    # pots_show = F.interpolate(pots_show, size=(256, 256), mode='bilinear', align_corners=True)
    # img_show = image
    # grid = [img, timg, (img - timg) * 5, pots_show]
    # grid_show = torch.cat(grid, dim=0)
    # grid_save = torchvision.utils.make_grid(grid_show, nrow=4, padding=5)
    # path = 'embeded.png'
    # torchvision.utils.save_image(grid_save, fp=path, nrow=4, padding=5)

    img_show = tensor2im(img)
    timg1_show = tensor2im(timg)
    gap_show = tensor2im(torch.abs(timg - img))
    noised_show = tensor2im(noised_img)
    pots_show = tensor2im(pots)
    pots_gt_show = tensor2im(pots_gt)
    warped_show = tensor2im(transformDec)

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
    img_ = cv2.cvtColor(noised_img, cv2.COLOR_RGB2BGR)
    folder_imgs = "embed_19_3_" + ".jpg"
    # cv2.imwrite(folder_imgs, img_, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
