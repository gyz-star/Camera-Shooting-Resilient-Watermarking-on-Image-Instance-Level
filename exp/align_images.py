from __future__ import print_function

import glob
import sys

import cv2
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt

import torch

sys.path.append("..")

MAX_FEATURES = 1000
GOOD_MATCH_PERCENT = 0.1


def alignImage(im1, im2):
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMINGLUT)
    matches = list(matcher.match(descriptors1, descriptors2, None))
    # matches = matcher.knnMatch(descriptors1, descriptors2, 2)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]
    # matches = matches[:4]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    assert len(points1) >= 4 and len(points2) >= 4
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape

    im1Reg = cv2.warpPerspective(im1, h, (width, height) ,borderValue=(140,149,94)) # , borderValue=(169, 169, 171)

    return im1Reg, h


def alignImages(refFolder, png_folder):
    """
    将拍摄图与参考图对齐
    流程是先将拍摄图大小resize为拍摄的1/6大小，再与参考图对齐后保存
    :param refFolder: 参考图，即生成的含密图
    :param png_folder: 拍摄图，即待对齐的图
    :return: 将对齐后的拍摄图保存
    """
    folders = os.listdir(refFolder)
    folders.sort()
    folders = folders[:-3]
    png_files = os.listdir(png_folder)
    png_files.sort()

    for i in range(len(folders)):
        folder_path = refFolder + folders[i] + "/"

        # Read reference image
        refFilename = folder_path + "container.png"

        print("Reading reference image : ", refFilename)
        imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)
        # imReference = cv2.resize(imReference, (384*2, 384*2))

        # Read image to be aligned
        imFilename = png_folder + "/" + png_files[i]
        print("Reading image to align : ", imFilename)
        im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
        h, w, _ = im.shape
        im = cv2.resize(im, (w // 6, h // 6))

        print("Aligning images ...")
        # Registered image will be resotred in imReg.
        # The estimated homography will be stored in h.
        imReg, h = alignImage(im, imReference)

        # Write aligned image to disk.
        outFilename = folder_path + "warped_screenshot.png"
        print("Saving aligned image : ", outFilename)
        # imReg = cv2.resize(imReg, (128*3, 128*3))
        cv2.imwrite(outFilename, imReg)

        # Print estimated homography
        print("Estimated homography : \n", h)


def deleteFiles(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 用于删除文件,,如果文件是一个目录则返回一个错误
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 递归删除一个目录以及目录内的所有内容
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def alignFolder(capturedFolder, encodedFolder, alignedFolder):
    captured_list = glob.glob(capturedFolder + '/*')
    if not os.path.exists(alignedFolder):
        os.makedirs(alignedFolder)
    for captured_path in captured_list:
        captured_name = captured_path.split('/')[-1].split('.')[0]
        tmp = captured_path.split('/')[-1].split('.')[1]
        encoded_path = encodedFolder + '/' + captured_name + '.' + tmp
        im_captured = cv2.imread(captured_path, cv2.IMREAD_COLOR)
        im_encoded = cv2.imread(encoded_path, cv2.IMREAD_COLOR)

        h, w, _ = im_captured.shape
        h = h // 3 if h > 1024 else h
        w = w // 3 if w > 1024 else w
        # im_captured = cv2.resize(im_captured, (w // 6, h // 6))
        im_captured = cv2.resize(im_captured, (w, h))
        im_aligned, H = alignImage(im_captured, im_encoded)
        save_path = alignedFolder + '/' + captured_name + '.' + tmp
        cv2.imwrite(save_path, im_aligned, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


if __name__ == '__main__':
    # imFilename = '/home/yizhi/Net/bizhi/newbizhi/19.jpg'
    imFilename = '/home/yizhi/Net/bizhi/xiuzheng2.jpg'
    # refFilename = '/home/yizhi/image_folder/Wild/Ours/global/encoded/8.jpg'
    # refFilename = '/home/yizhi/Net/image_folder/Wild/Ours/partial/encoded/8.jpg'
    refFilename = '/home/yizhi/Net/image_folder/Wild/origin/8.jpg'

    save_name = imFilename.split('/')[-1].split('.')[0]
    # outFilename = '/home/yizhi/image_folder/Wild/demo/a' + save_name + '.jpg'
    outFilename = '/home/yizhi/Net/bizhi/dagai/a' + save_name + '.jpg'
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (512, 512))
    h, w, _ = im.shape
    # im = cv2.resize(im, (w // 3, h // 3))
    # im = cv2.resize(im, (w, h))

    print(im.shape, imReference.shape)
    imReg, H = alignImage(im, imReference)
    # Write aligned image to disk.
    # imReg = cv2.resize(imReg, (128*3, 128*3))
    cv2.imwrite(outFilename, imReg)
    outfile=plt.imread(outFilename)
    plt.imshow(outfile)
    plt.axis('off')
    plt.show()


    # img_path = '/opt/data/mingjin/pycharm/Net/image_folder/Wild/print/RIHOOP/global'
    # ref_path = '/opt/data/mingjin/pycharm/Net/image_folder/Wild/RIHOOP/encoded'
    # save_path = '/opt/data/mingjin/pycharm/Net/image_folder/Wild/print/RIHOOP/aligned'
    # alignFolder(img_path, ref_path, save_path)
