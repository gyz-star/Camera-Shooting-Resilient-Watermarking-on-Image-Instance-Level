import datetime
import itertools
import os
import random

import cv2
import numpy as np
import torchgeometry
from dataset_deep_model import MyDataset
from dataset import myDataset
from model import DeepHomographyModel
from model import Encoder, Decoder, Discriminator, MaskRCNN
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import argparse
import time
# from points_rgb import genPN, template_embed, template_extract
from points_Ycbcr import genPN, template_embed_ycbcr, template_extract_ycbcr
from utils import get_cdos, get_warped_points_and_gap, get_rand_homography_mat, tensor2im, generate_random_key
from noise_layers.compose import Compose
from noise_layers.resizeThenBack import RESIZE
from args import options
import kornia as K
from exp.noise_layers_exp.jpeg import JPEGCompress
from noise import noise_layer
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
#训练几何校正

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', help='number of epochs to train', default=300, type=int)
    parser.add_argument('--lr', help='learning rate', default=0.0004, type=float)
    parser.add_argument('--image_size', default=400, type=int)
    parser.add_argument('--batch_size', default=4, help='batch size', type=int)
    parser.add_argument('--data_Train', default='/home/yizhi/Data/HiDDeN/train', help='root directory of the dataset', type=str)
    # parser.add_argument('--data_Train', default='/opt/data/mingjin/pycharm/Data/coco/train2017_2w', help='root directory of the dataset', type=str)
    parser.add_argument('--data_Val', default='/home/yizhi/Data/HiDDeN/val', help='root directory of the dataset', type=str)
    # parser.add_argument('--data_Train', default='/workshop/mingjin/pycharm/Data/Coco_mini/train', help='root directory of the dataset', type=str)
    # parser.add_argument('--data_Val', default='/workshop/mingjin/pycharm/Data/Coco_mini/val', help='root directory of the dataset', type=str)
    parser.add_argument('--K_size', type=int, default=7, help='size of PN')
    parser.add_argument('--warped_eps', type=float, default=0.2, help='hyper-parameter of noise')
    parser.add_argument('--noise', default='Vanilla', help='the noise to use when generating data',
                        choices=['Vanilla', 'Blur5', 'Blur10', 'Compression', "Gaussian", "S&P", "All"])
    parser.add_argument('--exp', type=str, default='NoNoise_2_4points_addEncoder', help='name of experiment')
    args = parser.parse_args()
    return args


def Train_model(args, model, criterion, optimizer, TrainLoader, ValLoader, startEpoch=0, tb_writer=None, key=None, encoder=None, maskRCNN=None):
    global time
    print("Starting Training")
    count = 0
    for epoch in range(startEpoch, args.epochs + 1):
        # print(epoch)
        # torch.cuda.empty_cache()
        for phase in ['train', 'val']:

            start_time = time.time()

            if phase == 'train':
                loader = TrainLoader
            else:
                loader = ValLoader

            testError = 0
            testLoss = 0
            for index, batch in enumerate(loader):
                optimizer.zero_grad()
                st_time = time.time()
                ####
                item_img = batch['img'].to(device)
                item_sec = batch['sec'].to(device)

                with torch.no_grad():
                    mask = maskRCNN(item_img)
                if mask is None:
                    continue
                with torch.no_grad():
                    encoded = encoder(item_img, item_sec)
                    result = encoded * mask + item_img * (1 - mask)

                item_ones = torch.ones_like(item_img).float().to(device)

                source_points = get_cdos(item_img, 4)
                pn1 = genPN(args.K_size)
                t_img, PN1 = template_embed_ycbcr(result, source_points, PN=pn1, )
                t_ones, PN2 = template_embed_ycbcr(item_ones, source_points, PN=pn1, )

                # 加噪
                # noised_img = noise_layer(t_img, key=key)

                # 透视扭曲
                noised_img = t_img
                h, w = noised_img.shape[2:4]
                homography = get_rand_homography_mat(t_img, args.warped_eps)
                homography = torch.from_numpy(homography).float().to(device)
                transformDec = torchgeometry.warp_perspective(noised_img, homography[:, 1, :, :], dsize=(h, w))

                #### inputs
                pn2 = genPN(args.K_size)
                Ip1 = template_extract_ycbcr(transformDec, pn2).to(device)
                Ip2 = template_extract_ycbcr(t_ones, pn2).to(device)

                inputs = torch.cat([Ip1, Ip2], dim=1)

                #### target
                target, _ = get_warped_points_and_gap(source_points, homography)

                outputs = model.forward(inputs)
                loss = criterion(outputs, target.view(-1, 8))
                testError += loss.item() ** 0.5
                testLoss += loss.item()

                # elapsed_time = time.time() - st_time
                # print(elapsed_time)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    count += 1
                    if tb_writer is not None and (count % 500 == 0 or count == 1):
                        with torch.no_grad():
                            item_img_grid = vutils.make_grid(item_img[0:2, ...], nrow=5)
                            t_img_grid = vutils.make_grid(t_img[0:2, ...], nrow=5)
                            se_grid = vutils.make_grid(t_img[0:2, ...] - item_img[0:2, ...], nrow=5)
                            t_ones_grid = vutils.make_grid(t_ones[0:2, ...], nrow=5)
                            noised_img_grid = vutils.make_grid(noised_img[0:2, ...], nrow=5)
                            transformDec_grid = vutils.make_grid(transformDec[:5, ...], nrow=5)
                            Ip1_grid = vutils.make_grid(Ip1[0:2, ...], nrow=5)
                            Ip2_grid = vutils.make_grid(Ip2[0:2, ...], nrow=5)

                            tb_writer.add_image("item_img", item_img_grid, count)
                            tb_writer.add_image("t_img", t_img_grid, count)
                            tb_writer.add_image("se_grid", se_grid, count)
                            tb_writer.add_image("t_ones", t_ones_grid, count)
                            tb_writer.add_image("noised_img", noised_img_grid, count)
                            tb_writer.add_image("transformDec", transformDec_grid, count)
                            tb_writer.add_image("Ip1", Ip1_grid, count)
                            tb_writer.add_image("Ip2", Ip2_grid, count)

                            # item_img_grid = tensor2im(item_img[8].unsqueeze(0).clone())
                            # t_img_grid = tensor2im(t_img[8].unsqueeze(0).clone())
                            # se_grid = tensor2im(t_img[8].unsqueeze(0).clone() - item_img[8].unsqueeze(0).clone())
                            # t_ones_grid = tensor2im(t_ones[8].unsqueeze(0).clone())
                            # noised_img_grid = tensor2im(noised_img[8].unsqueeze(0).clone())
                            # transformDec_grid = tensor2im(transformDec[8].unsqueeze(0).clone())
                            # Ip1_grid = tensor2im(Ip1[8].unsqueeze(0))
                            # Ip2_grid = tensor2im(Ip2[8].unsqueeze(0))

                            # tb_writer.add_image("item_img", item_img_grid, count, dataformats='HWC')
                            # tb_writer.add_image("t_img", t_img_grid, count, dataformats='HWC')
                            # tb_writer.add_image("se_grid", se_grid, count, dataformats='HWC')
                            # tb_writer.add_image("t_ones", t_ones_grid, count, dataformats='HWC')
                            # tb_writer.add_image("noised_img", noised_img_grid, count, dataformats='HWC')
                            # tb_writer.add_image("transformDec", transformDec_grid, count, dataformats='HWC')
                            # tb_writer.add_image("Ip1", Ip1_grid, count, dataformats='HW')
                            # tb_writer.add_image("Ip2", Ip2_grid, count, dataformats='HW')

                del loss
                del outputs
                del item_img
                del target

            cornerAvgError = testError / len(loader)
            avgLoss = testLoss / len(loader)
            elapsed_time = time.time() - start_time
            elapsed_time_ = str(elapsed_time).split('.')[0]
            print('**********************')
            print('Phase: ' + str(phase))
            print('Epoch Number: [{}/{}] | Corner Avg Error (CAE): {:.6f}, Loss : {:.6f}'.format(epoch, args.epochs, cornerAvgError, avgLoss))
            print('Time elapsed: ' + str(elapsed_time_) + ' seconds')
            time_ = datetime.datetime.now()
            time_str = str(time_).split('.')[0]
            print('Now: ' + time_str)

            if tb_writer is not None:
                tb_writer.add_scalar('%s/ACE' % str(phase), float(cornerAvgError), epoch)
                tb_writer.add_scalar('%s/Loss' % str(phase), float(avgLoss), epoch)

        if epoch % 50 == 0:
            state = {
                'model': model.state_dict(),
                'next_epoch': epoch + 1
            }
            save_path = './output/deep_model/%s/' % args.exp
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(state, save_path + str(epoch) + '.pth')

    del TrainLoader
    del ValLoader


def main():
    args = parseArgs()
    tb_writer = SummaryWriter(logdir='./log/tensorboard/%s' % args.exp)
    # tb_writer = None
    model = DeepHomographyModel().to(device)

    maskrcnn = MaskRCNN().to(device)
    encoder = Encoder(message_length=30).to(device)

    # md_path = '/opt/data/mingjin/pycharm/MyFirstNet/output/6_DDHNewAdjust5_Parameter/1139.pth'
    # model_pth = torch.load(md_path, map_location=torch.device('cpu'))
    # encoder.load_state_dict(model_pth['model_encoder'])
    encoder.eval()
    maskrcnn.eval()

    key = generate_random_key(10)
    opt = options.getOpt()
    # compose_noise = Compose(opt, None).to(device)
    start_epoch = 1

    # cp_path = '/opt/data/mingjin/pycharm/Net/output/deep_model/AllNoise_2_4points/50.pth'
    # checkpoint = torch.load(cp_path, map_location=torch.device('cpu'))
    # # start_epoch = checkpoint['next_epoch']
    # start_epoch = 51
    # model.load_state_dict(checkpoint['model'])

    print("Arguments are:")
    print(args)
    print("------------------")

    trainData = MyDataset(args.data_Train, args.image_size)
    valData = MyDataset(args.data_Val, args.image_size)

    TrainLoader = DataLoader(trainData, args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    ValLoader = DataLoader(valData, 1, shuffle=True, num_workers=8, drop_last=True)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))

    Train_model(args, model, criterion, optimizer, TrainLoader, ValLoader, startEpoch=start_epoch, tb_writer=tb_writer, key=key, encoder=encoder,
                maskRCNN=maskrcnn)
    if tb_writer is not None:
        tb_writer.close()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True  # False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main()
