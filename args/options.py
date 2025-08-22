# !/usr/bin/env python
# -- coding: utf-8 --
# 小何一定行
import argparse


def getOpt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs of training')
    parser.add_argument('--batchSize', type=int, default=4, help='size of the batches')
    parser.add_argument('--datarootTrain', type=str, default='/opt/data/mingjin/pycharm/Data/HiDDeN/train', help='root directory of the dataset')
    parser.add_argument('--datarootTest', type=str, default='/opt/data/mingjin/pycharm/Data/HiDDeN/val', help='root directory of the dataset')
    parser.add_argument('--lr_G', type=float, default=0.004, help='initial learning rate for G')
    parser.add_argument('--lr_D', type=float, default=0.001, help='initial learning rate for D')
    parser.add_argument('--image_size', type=int, default=256, help='size of the image')
    parser.add_argument('--extract_size', type=int, default=400, help='size of the image')
    parser.add_argument('--message_length', type=int, default=30, help='size of the message')
    parser.add_argument('--device_ID', type=str, default='cuda:3', help='the used device ID')
    parser.add_argument('--vis_iter', type=int, default=200, help='Iterate a certain number of times for visualization')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--exp', type=str, default='retrain the previous model_1', help='name of experiment')
    # ---------------------------------------------------------------------------------------------------------------- #
    # ----- 噪声超参 ----- #
    parser.add_argument('--rnd_trans_ramp', type=int, default=15000, help='hyper-parameter of noise')
    parser.add_argument('--rnd_bri_ramp', type=int, default=5000, help='hyper-parameter of noise')
    parser.add_argument('--rnd_sat_ramp', type=int, default=5000, help='hyper-parameter of noise')
    parser.add_argument('--rnd_hue_ramp', type=int, default=5000, help='hyper-parameter of noise')
    parser.add_argument('--rnd_noise_ramp', type=int, default=5000, help='hyper-parameter of noise')
    parser.add_argument('--contrast_ramp', type=int, default=5000, help='hyper-parameter of noise')
    parser.add_argument('--jpeg_quality_ramp', type=int, default=5000, help='hyper-parameter of noise')

    parser.add_argument('--rnd_trans', type=float, default=0.2, help='hyper-parameter of noise')
    parser.add_argument('--rnd_noise', type=float, default=0.02, help='hyper-parameter of noise')
    parser.add_argument('--rnd_bri', type=float, default=0.3, help='hyper-parameter of noise')
    parser.add_argument('--rnd_sat', type=float, default=1.0, help='hyper-parameter of noise')
    parser.add_argument('--rnd_hue', type=float, default=0.1, help='hyper-parameter of noise')
    parser.add_argument('--contrast_low', type=float, default=0.5, help='hyper-parameter of noise')
    parser.add_argument('--contrast_high', type=float, default=1.5, help='hyper-parameter of noise')
    parser.add_argument('--jpeg_quality', type=int, default=50, help='hyper-parameter of noise')
    parser.add_argument('--resize_min', type=float, default=0.5, help='hyper-parameter of noise')
    parser.add_argument('--resize_max', type=float, default=2.0, help='hyper-parameter of noise')

    # ----- loss权重超参 ----- #
    parser.add_argument('--img_loss_ramp', type=int, default=15000, help='hyper-parameter of loss')

    opt = parser.parse_args()

    return opt
