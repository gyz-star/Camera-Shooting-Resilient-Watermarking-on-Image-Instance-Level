# !/usr/bin/env python
# -- coding: utf-8 --

import cv2
import numpy as np
import pywt
import torch
from scipy import signal
from matplotlib import pyplot as plt
from operator import itemgetter, attrgetter
from scipy import ndimage
from skimage.transform import radon, rotate
import torchvision.transforms as T
import torchvision
from PIL import Image

from Net.utils import get_cdos, min_max_scale, get_rand_homography_mat


def genPN(sz):
    rd = np.random.RandomState(1)
    PN = rd.randint(0, 2, [sz, sz]).astype('double') * 2 - 1
    return PN


def template_embed(img, pots, PN=genPN(16)):
    scale = 4  # Magnification of correlation

    # shape value of image should be even
    if np.mod(img.shape[0], 2) == 1:
        img = np.concatenate((np.reshape(img[1, :, :], [1, img.shape[1], img.shape[2]]), img), axis=0)
    if np.mod(img.shape[1], 2) == 1:
        img = np.concatenate((np.reshape(img[:, 1, :], [img.shape[0], 1, img.shape[2]]), img), axis=1)

    PNwid = PN.shape[0]
    cA, (cH, cV, cD) = pywt.dwt2(img[:, :, 2], 'db2')
    # cD1 = np.copy(cD)

    cov = cD
    R = signal.convolve2d(pow(cov, 2), PN, mode='same', boundary='symm')
    t = R.max() * scale  # threshold
    z1 = np.zeros(pots.shape[0])
    for i in range(pots.shape[0]):
        vetx = cov[pots[i, 0]:pots[i, 0] + PNwid, pots[i, 1]:pots[i, 1] + PNwid]
        z = np.sum(pow(vetx, 2) * PN)  # z=x^T*W*x
        a = np.sum(pow(vetx, 2))  # a=x^T*x
        if (z >= 0):
            # (1+alpha^2)*z + 2*alpha*a > threshold
            z = 1e-10 if z == 0 else z
            # alpha = (np.sqrt(a * a - z * z + z * t) - a) / z
            alpha = 1000
            # alpha = 700
            vetx = vetx * (1.0 + alpha * PN)
        else:
            # (1+alpha^2)*z - 2*alpha*a < -threshold
            # alpha = (-np.sqrt(a * a - z * z - z * t) + a) / z
            alpha = -1000
            # alpha = 700
            vetx = vetx * (1.0 - alpha * PN)
        cov[pots[i, 0]:pots[i, 0] + PNwid, pots[i, 1]:pots[i, 1] + PNwid] = vetx
        z1[i] = np.sum(pow(vetx, 2) * PN)

    wdimg = np.array([img[:, :, 0], img[:, :, 1], pywt.idwt2((cA, (cH, cV, cD)), 'db2')])
    wdimg = wdimg.transpose(1, 2, 0).astype('uint8')
    return wdimg, PN


def template_extract(img, PN):
    PNwid = PN.shape[0]
    cA, (cH, cV, cD) = pywt.dwt2(img[:, :, 2], 'db2')
    cov = cD
    # G = np.zeros(img[:, :, 2].shape)
    # G[0:PNwid, 0:PNwid] = PN
    rst = signal.convolve2d(pow(cov, 2), PN, mode='same', boundary='symm')
    plt.imshow(rst)
    return rst


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # imgpath = 'E:/ImageDataBase/BOWS2OrigEp3/0.pgm'
    # imgpath = 'E:/ImageDataBase/1.jpg'
    # imgpath = '/opt/data/mingjin/pycharm/MyFirstNet/sample/origin/8.jpg'
    imgpath = '/opt/data/mingjin/pycharm/MyFirstNet/sample/origin/1.jpg'
    image = cv2.imread(imgpath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))
    # cdos1 = np.array([[100, 100], [100, 150], [150, 150], [150, 100]])
    # cdos1 = np.array([[20, 20], [20, 50], [50, 20], [50, 50]])
    cdos1 = get_cdos(torch.rand(1, 3, 256, 256))
    timg, PN = template_embed(image, cdos1, PN=genPN(7))

    # img_ = cv2.cvtColor(timg, cv2.COLOR_RGB2BGR)
    # folder_imgs = "1" + ".jpg"
    # cv2.imwrite(folder_imgs, img_, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    #
    # img_ = cv2.imread(folder_imgs)
    # timg_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)

    pots = template_extract(timg, PN)
    pots = min_max_scale(pots)

    homography = get_rand_homography_mat(np.array(torch.rand(1, 3, 256, 256)), 0.2)
    warped = cv2.warpPerspective(timg, homography[0][1], (256, 256))

    pots_warp = template_extract(warped, PN)
    pots_warp = min_max_scale(pots_warp)

    plt.figure(1)
    plt.subplot(2, 3, 1)
    plt.imshow(image)
    plt.subplot(2, 3, 2)
    plt.imshow(timg)
    plt.subplot(2, 3, 3)
    plt.imshow(timg - image)
    plt.subplot(2, 3, 4)
    plt.axis('off')
    plt.imshow(pots, cmap=plt.cm.gray)
    plt.subplot(2, 3, 5)
    plt.axis('off')
    plt.imshow(pots_warp, cmap=plt.cm.gray)
    plt.show()
