# -*- coding:utf-8 -*-
# 小何一定行
# !/usr/bin/python3

import argparse
import datetime
import itertools
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import numpy as np
# import torchgeometry
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from tensorboardX import SummaryWriter
from pytorch_msssim import MS_SSIM, SSIM#pytorch要1.7
import kornia
import torchvision.utils as vutils
# from torchsummary import summary
# import lpips
# from piqa import LPIPS
from Lpips.lpips_pytorch import LPIPS, lpips

from model import Encoder, Decoder, Discriminator, MaskRCNN
from utils import same_seeds, get_secret_acc, get_warped_extracted_IMG, weights_init
from vgg_loss import VGGLoss
from dataset import myDataset
from noise_layers.compose import Compose  # 将噪声分组
from noise_layers.composeTest import ComposeTest  # 依次叠加噪声
from args import options

####################################################################################################################
opt = options.getOpt()
tb_writer = SummaryWriter(logdir='./log/%s' % opt.exp)
same_seeds(42)  # 固定 random seed 的函數，以便 reproduce。
device = torch.device(opt.device_ID)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
# -----Definition of variables----- #
# Networks
maskrcnn = MaskRCNN().to(device)
encoder = Encoder(message_length=opt.message_length).to(device)
decoder = Decoder(message_length=opt.message_length).to(device)
discriminator = Discriminator().to(device)
encoder.apply(weights_init)
decoder.apply(weights_init)
discriminator.apply(weights_init)

# Noise layers
compose = Compose(opt, tb_writer).to(device)
composeTest = ComposeTest(opt).to(device)

# Losses
criterion_L1 = torch.nn.L1Loss().to(device)
criterion_L2 = torch.nn.MSELoss().to(device)
criterion_SEC = torch.nn.BCELoss().to(device)
criterion_GAN = torch.nn.MSELoss().to(device)
criterion_VGG = VGGLoss(3, 1, False).to(device)
computeSSIM = SSIM(1.0).to(device)
computeMSSSIM = MS_SSIM(1.0).to(device)
# criterion_LPIPS = lpips.LPIPS(net='vgg', verbose=False).to(device)
criterion_LPIPS = LPIPS().to(device)
# criterion_LPIPS = LPIPS(net_type='alex', version='0.1').to(device)

PSNR_Enc_iter, SSIM_Enc_iter, bit_iter = [], [], []
losses_iter, loss_sec_iter, loss_img_iter = [], [], []
loss_D_iter, loss_G_iter = [], []

PSNR_Enc_epoch, SSIM_Enc_epoch, bit_epoch = [], [], []
losses_epoch, loss_sec_epoch, loss_img_epoch = [], [], []
loss_D_epoch, loss_G_epoch = [], []

# Optimizers & LR schedulers //itertools.chain(encoder.parameters(), decoder.parameters())
# param = [{"params": encoder.parameters()},
#          {"params": decoder.parameters()}, ]
optimizer_SEC = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=opt.lr_G, betas=(0.5, 0.999))
optimizer_IMG = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=opt.lr_G, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr_D, betas=(0.5, 0.999))
# Dataset loader
dataloaderTrain = DataLoader(myDataset(opt, True), batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu, drop_last=True)
dataloaderTest = DataLoader(myDataset(opt, False), batch_size=1, shuffle=False, num_workers=opt.n_cpu, drop_last=True)

# 加载checkpoint模型
# cp_path = '/home/yizhi/Net/output/retrain the previous model_1/1207.pth'
# checkpoint = torch.load(cp_path, map_location=torch.device('cpu'))
# encoder.load_state_dict(checkpoint['model_encoder'])
# decoder.load_state_dict(checkpoint['model_decoder'])
# discriminator.load_state_dict(checkpoint['model_discriminator'])
#
# glob_step = checkpoint['glob_step']
# star_epoch = checkpoint['epoch']
# noise_step = checkpoint['noise_step']
# img_loss_step = checkpoint['img_loss_step']
glob_step = 1
noise_step = 1
tps_step = 1
img_loss_step = 1
star_epoch = opt.epoch
# ---------------------------------------------------------------------------------------------------------------- #
###################################
# print(encoder)
# print(decoder)
# print(discriminator)
# summary(decoder, (3, 256, 256))
###################################
print(opt)
print('train_len:', len(dataloaderTrain), 'test_len:', len(dataloaderTest))
print('epoch_star:', star_epoch, 'step_star:', glob_step, )
print('noise_step:', noise_step, 'img_loss_step:', img_loss_step)
####################################################################################################################
train_bit = 0
for epoch in range(star_epoch, star_epoch + opt.n_epochs):
    # ----- Training-----#
    maskrcnn.eval()
    encoder.train()
    # encoder.eval()
    decoder.train()
    discriminator.train()
    # discriminator.eval()
    ###################################
    train_img = True  # 图像质量开关
    use_gan = True  # 是否加入gan网络
    use_noise = False
    ###################################
    for i, batch in enumerate(dataloaderTrain):
        realImage, realSecret = batch['img'], batch['sec']
        realImage, realSecret = realImage.to(device), realSecret.to(device)

        # ----- Network Pipeline -----#
        with torch.no_grad():
            mask = maskrcnn(realImage)
        if mask is None:
            continue
        # with torch.no_grad():
        encoded = encoder(realImage, realSecret)
        se = (encoded - realImage)
        result = encoded * mask + realImage * (1 - mask)
        noised_img, noised_mask = result, mask
        ################################################################
        if use_noise:
            if epoch > 20:  # 55
                # 膨胀
                f = torch.from_numpy(np.ones((3, 3, 4, 4), dtype=np.float32)).to(device)
                noised_mask = F.conv2d(mask, f, bias=None, padding=3, dilation=2)
                noised_mask[noised_mask > 0] = 1.0
            if epoch > 30:  # 71
                # 各种噪声
                noised_img = compose(noised_img, noise_step)
                noise_step += 1
        # ---------------------------------------------------------------------------------------------------------------------------#
        dec = get_warped_extracted_IMG(noised_img.clone(), noised_mask.clone(), args=opt, isMul=True, isHomography=True, isResize=True)
        if dec is None:
            continue
        # ---------------------------------------------------------------------------------------------------------------------------#
        pred_bit = decoder(dec)
        pred_bit = torch.tensor(pred_bit)
        # print(pred_bit)
        # print(realSecret)
        # 消息提取loss
        SEC = criterion_SEC(pred_bit, realSecret)
        if train_img:
            if use_gan:
                # ----- Discriminator -----#
                pred_real = discriminator(realImage)
                pred_fake = discriminator(encoded.detach())
                # Real loss
                loss_D_real = criterion_GAN(pred_real, torch.ones(pred_real.size()).to(device))
                # Fake loss
                loss_D_fake = criterion_GAN(pred_fake, torch.zeros(pred_fake.size()).to(device))
                # Total loss
                loss_D = (loss_D_real + loss_D_fake) * 0.5

                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()
            #######################################
            # ----- Generators and Extractor -----#
            # 图像生成Loss
            vgg_on_cov = criterion_VGG(realImage)
            vgg_on_enc = criterion_VGG(encoded)
            VGG = criterion_L2(vgg_on_cov, vgg_on_enc)
            # LPIPS = torch.mean(criterion_LPIPS(encoded, realImage, normalize=True))
            LPIPS = criterion_LPIPS(encoded, realImage)
            L2 = criterion_L2(encoded, realImage)
            L1 = criterion_L1(encoded, realImage)
            ssim = 1 - computeSSIM(encoded, realImage)
            L1SSIM_loss = 0.16 * L1 + 0.84 * ssim
            # 0.7 * vgg_loss + 2 * sec_loss + 0.6 * (0.16 * l1_loss + 0.84 * ssim_loss)
            # IMG = 0.7 * VGG + 0.6 * L1SSIM_loss
            # IMG = VGG * 1.5 + L1SSIM_loss * 0.6 + LPIPS * 1.5
            # IMG = VGG * 1.0 + L2 * 2 + L1 * 5 + ssim * 0.5  # VGG
            # IMG = VGG * 1.0 + LPIPS * 1.0 + ssim * 1.0  # RIHOOP + L2 * 5.0
            IMG = VGG * 1.0 + LPIPS * 1.0 + L2 * 5 + ssim * 0.5  # 论文
            # IMG = VGG * 1.0 + LPIPS * 1.0 + L2 * 5 + (0.16 * L1 + 0.84 * ssim) * 0.5

            # VGG * 0.8 + L2 * 2.5 + L1 * 5 + ssim * 0.1 + LSGAN * 0.005
            # IMG = VGG * 0.7 + L2 * 2.5 + L1 * 10 + ssim * 0.5
            ############################################################################

            IMG_scale = min(1.0 * img_loss_step / opt.img_loss_ramp, 1.0)
            # IMG_scale = 1

            loss_G = SEC
            # loss_G = IMG * IMG_scale + SEC
            # loss_G = IMG * IMG_scale + SEC * 3
            # loss_G = IMG + SEC * 2
            # loss_G = IMG + SEC
            # if glob_step % opt.vis_iter == 0 or glob_step == 1:
            #     print(IMG_scale)

            if use_gan:
                pred_fake = discriminator(encoded)
                loss_G_fake = criterion_GAN(pred_fake, torch.ones(pred_fake.size()).to(device))
                loss_G = loss_G + loss_G_fake * 0.001 * IMG_scale
            img_loss_step += 10
            optimizer_IMG.zero_grad()
            loss_G.backward()
            optimizer_IMG.step()
        else:
            # Total loss
            loss_G = SEC

            optimizer_SEC.zero_grad()
            loss_G.backward()
            optimizer_SEC.step()
        ############################################################################
        # ----- Record of indicators -----#
        if glob_step % (5 * opt.vis_iter) == 0 or glob_step == 1:
            with torch.no_grad():
                origin_img_grid = vutils.make_grid(realImage[:5, ...], nrow=5)
                origin_mask_grid = vutils.make_grid(mask[:5, ...], nrow=5)
                se_grid = vutils.make_grid(se[:5, ...], nrow=5)
                encoded_grid = vutils.make_grid(encoded[:5, ...], nrow=5)
                result_grid = vutils.make_grid(result[:5, ...], nrow=5)
                noised_grid = vutils.make_grid(noised_img[:5, ...], nrow=5)
                noisedMask_grid = vutils.make_grid(noised_mask[:5, ...], nrow=5)
                dec_grid = vutils.make_grid(dec[:5, ...], nrow=5)

                tb_writer.add_image("image", origin_img_grid, glob_step)
                tb_writer.add_image("mask", origin_mask_grid, glob_step)
                tb_writer.add_image("se", se_grid, glob_step)
                tb_writer.add_image("result", result_grid, glob_step)
                tb_writer.add_image("encoded", encoded_grid, glob_step)
                tb_writer.add_image("noised", noised_grid, glob_step)
                tb_writer.add_image("noisedMask", noisedMask_grid, glob_step)
                tb_writer.add_image("dec", dec_grid, glob_step)
        with torch.no_grad():
            # 消息提取
            bit_acc = get_secret_acc(pred_bit.detach(), realSecret.detach())
            # decoded_rounded = pred_bit.detach().cpu().numpy().round().clip(0, 1)
            # bit_acc = 1 - np.sum(np.abs(decoded_rounded - realSecret.detach().cpu().numpy())) / (opt.batchSize * realSecret.shape[1])
            # 图像质量
            batch_enc_ssim = computeSSIM(encoded, realImage)
            batch_enc_psnr = kornia.losses.psnr(encoded, realImage, 1)

            # if glob_step % 10 == 0 or glob_step == 1:
            #     print('step|epoch:%d|%d,bit_acc:%s,psnr:%s,ssim:%s' % (glob_step, epoch, str(bit_acc), str(batch_enc_psnr), str(batch_enc_ssim)))

            SSIM_Enc_iter.append(batch_enc_ssim.item())
            PSNR_Enc_iter.append(batch_enc_psnr.item())
            losses_iter.append(loss_G.item())
            loss_sec_iter.append(SEC.item())
            bit_iter.append(float(bit_acc))
            if train_img:
                loss_img_iter.append(IMG.item())
                if use_gan:
                    loss_D_iter.append(loss_D.item())
                    loss_G_iter.append(loss_G_fake.item())

            SSIM_Enc_epoch.append(batch_enc_ssim.item())
            PSNR_Enc_epoch.append(batch_enc_psnr.item())
            losses_epoch.append(loss_G.item())
            loss_sec_epoch.append(SEC.item())
            bit_epoch.append(float(bit_acc))
            if train_img:
                loss_img_epoch.append(IMG.item())
                if use_gan:
                    loss_D_epoch.append(loss_D.item())
                    loss_G_epoch.append(loss_G_fake.item())

        if glob_step % opt.vis_iter == 0 or glob_step == 1:
            tb_writer.add_scalar('iter_metric/encoded_PSNR', float(np.mean(PSNR_Enc_iter)), glob_step)
            tb_writer.add_scalar('iter_metric/encoded_SSIM', float(np.mean(SSIM_Enc_iter)), glob_step)

            tb_writer.add_scalar('iter_metric/losses', float(np.mean(losses_iter)), glob_step)
            tb_writer.add_scalar('iter_metric/loss_sec', float(np.mean(loss_sec_iter)), glob_step)
            tb_writer.add_scalar('iter_metric/bit_acc', float(np.mean(bit_iter)), glob_step)
            if train_img:
                tb_writer.add_scalar('iter_metric/loss_img', float(np.mean(loss_img_iter)), glob_step)
                if use_gan:
                    tb_writer.add_scalars('iter_metric/loss_adv', {'D': float(np.mean(loss_D_iter)),
                                                                   'G': float(np.mean(loss_G_iter))}, glob_step)
            ############################################################################
        if glob_step % opt.vis_iter == 0 or glob_step == 1:
            time = datetime.datetime.now()
            time_str = str(time).split('.')[0]
            print('---------- 当前时间 : %s ----------' % time_str)
            print('step:%d|epoch:%d,%d/%d|losses:%.3f|psnr:%.2f,bit_acc:%.4f' % (
                glob_step, epoch + 1, i + 1, len(dataloaderTrain), float(np.mean(losses_iter)),
                float(np.mean(PSNR_Enc_iter)), float(np.mean(bit_iter))))

            ############################################################################
            PSNR_Enc_iter, SSIM_Enc_iter, bit_iter = [], [], []
            losses_iter, loss_sec_iter, loss_img_iter = [], [], []
            loss_D_iter, loss_G_iter = [], []
        ##################################################################################################################################
        glob_step += 1
    ##################################################################################################################################
    with torch.no_grad():
        tb_writer.add_scalar('epoch_metric/encoded_PSNR', np.mean(PSNR_Enc_epoch), epoch + 1)
        tb_writer.add_scalar('epoch_metric/encoded_SSIM', np.mean(SSIM_Enc_epoch), epoch + 1)

        tb_writer.add_scalar('epoch_metric/losses', np.mean(losses_epoch), epoch + 1)
        tb_writer.add_scalar('epoch_metric/loss_sec', np.mean(loss_sec_epoch), epoch + 1)
        tb_writer.add_scalar('epoch_metric/bit_acc', np.mean(bit_epoch), epoch + 1)
        if train_img:
            tb_writer.add_scalar('epoch_metric/loss_img', np.mean(loss_img_epoch), epoch + 1)
            if use_gan:
                tb_writer.add_scalars('epoch_metric/loss_adv', {'D': float(np.mean(loss_D_epoch)),
                                                                'G': float(np.mean(loss_G_epoch))}, epoch + 1)

        PSNR_Enc_epoch, SSIM_Enc_epoch, bit_epoch = [], [], []
        losses_epoch, loss_sec_epoch, loss_img_epoch = [], [], []
        loss_D_epoch, loss_G_epoch = [], []

    # Save models checkpoints
    state = {
        'model_encoder': encoder.state_dict(),
        'model_decoder': decoder.state_dict(),
        'model_discriminator': discriminator.state_dict(),
        'epoch': epoch + 1,
        'glob_step': glob_step,
        'noise_step': noise_step,
        'img_loss_step': img_loss_step,
    }
    save_path = './output/%s/' % opt.exp
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(state, save_path + '%d.pth' % (epoch + 1))

    # ----------------------------------------------------------------------------------------------------- #
    # teat the model #
    maskrcnn.eval()
    encoder.eval()
    decoder.eval()

    PSNR_Enc_val, SSIM_Enc_val, bit_val = [], [], []

    for i, batch in enumerate(dataloaderTest):
        realImage, realSecret = batch['img'], batch['sec']
        realImage, realSecret = realImage.to(device), realSecret.to(device)
        with torch.no_grad():
            encoded = encoder(realImage, realSecret)
            mask = maskrcnn(realImage)
            if mask is None:
                continue

            result = encoded * mask + realImage * (1 - mask)

            noised_img, noised_mask = result, mask
            if use_noise:
                if epoch > 20:  # 50
                    # 膨胀
                    f = torch.from_numpy(np.ones((3, 3, 4, 4), dtype=np.float32)).to(device)
                    noised_mask = F.conv2d(mask, f, bias=None, padding=3, dilation=2)
                    noised_mask[noised_mask > 0] = 1.0
                if epoch > 30:
                    noised_img = composeTest(noised_img, noise_step)

            noised_img = (255.0 * torch.clamp(noised_img, 0.0, 1.0)).long()
            noised_img = noised_img.float() / 255.0

            decs = get_warped_extracted_IMG(noised_img, noised_mask, opt, isMul=True, isHomography=True, isResize=True)
            pred_bit = decoder(decs)

            bit_acc = get_secret_acc(pred_bit, realSecret)
            batch_enc_ssim = computeSSIM(encoded, realImage)
            batch_enc_psnr = kornia.losses.psnr(encoded, realImage, 1)

            SSIM_Enc_val.append(batch_enc_ssim.item())
            PSNR_Enc_val.append(batch_enc_psnr.item())
            bit_val.append(float(bit_acc))

            tb_writer.add_scalar('val_metric/encoded_PSNR', np.mean(PSNR_Enc_val), epoch + 1)
            tb_writer.add_scalar('val_metric/encoded_SSIM', np.mean(SSIM_Enc_val), epoch + 1)
            tb_writer.add_scalar('val_metric/bit_acc', np.mean(bit_val), epoch + 1)

    test_bit = np.mean(bit_val)

tb_writer.close()
###################################
