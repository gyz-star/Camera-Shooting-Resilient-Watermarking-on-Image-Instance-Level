# !/usr/bin/env python
# -- coding: utf-8 --
# 小何一定行
import cv2
import numpy as np
import random
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import torchgeometry
from torchvision import models
from noise_layers.compose import Compose
from matplotlib import pyplot as plt
from utils import find_border,near_border

class Encoder(nn.Module):
    def __init__(self, ngf=64, message_length=30, add_image=True):
        super(Encoder, self).__init__()
        print("train on dense net...")
        self.data_depth = message_length
        self.hidden_size = ngf
        self._models = self._build_models()
        self.add_image = add_image

    def forward(self, image, message):
        # 消息矩阵
        img = image - 0.5
        sec = message - 0.5
        h, w = img.shape[2], img.shape[3]

        expanded_message = sec.unsqueeze(-1).unsqueeze(-1)#扩充消息矩阵
        expanded_message = expanded_message.expand(-1, -1, h, w)

        x = self._models[0](img)
        x_list = [x]

        for layer in self._models[1:]:
            x = layer(torch.cat(x_list + [expanded_message], dim=1))
            x_list.append(x)

        if self.add_image:
            x = image + x

        return torch.sigmoid(x)

    def _build_models(self):
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.hidden_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.hidden_size),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.hidden_size),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.hidden_size),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.hidden_size),
            nn.LeakyReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.hidden_size + self.data_depth, self.hidden_size, kernel_size=3, padding=1, bias=False),#hiiden-size为64，depth是嵌入消息长度，压缩成64
            nn.BatchNorm2d(self.hidden_size),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.hidden_size),
            nn.LeakyReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.hidden_size * 2 + self.data_depth, self.hidden_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.hidden_size),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.hidden_size),
            nn.LeakyReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.hidden_size * 3 + self.data_depth, 3, kernel_size=3, padding=1)
        )

        return self.conv1, self.conv2, self.conv3, self.conv4


class PreModel(nn.Module):
    def __init__(self, ngf=64, add_image=False):
        super(PreModel, self).__init__()
        print("train on dense net...")
        self.hidden_size = ngf
        self._models = self._build_models()
        self.add_image = add_image

    def forward(self, image):
        # 消息矩阵
        img = image - 0.5
        h, w = img.shape[2], img.shape[3]

        x = self._models[0](img)
        x_list = [x]

        for layer in self._models[1:]:
            x = layer(torch.cat(x_list, dim=1))
            x_list.append(x)

        if self.add_image:
            x = image + x

        return x

    def _build_models(self):
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.hidden_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.hidden_size),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.hidden_size),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.hidden_size),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.hidden_size),
            nn.LeakyReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.hidden_size),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.hidden_size),
            nn.LeakyReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.hidden_size * 2, self.hidden_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.hidden_size),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.hidden_size),
            nn.LeakyReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.hidden_size * 3, 1, kernel_size=3, padding=1)
        )

        return self.conv1, self.conv2, self.conv3, self.conv4


class Decoder(nn.Module):
    def __init__(self, ndf=64, message_length=30):
        super(Decoder, self).__init__()

        self.ndf = ndf
        self.message_length = message_length
        self.subNetwork1 = SubNetwork1(ndf=self.ndf)  # 特征提取网络->3卷积块+2残差块
        self.subNetwork2 = SubNetwork2(ndf=self.ndf)  # 注意力自网络->1卷积块+5残差块+1个3*3卷积降维
        self.subNetwork3 = SubNetwork3(ndf=self.ndf,
                                       message_length=self.message_length)  # 解码子网络->1个卷积块+7个残差块+1*1卷积降维+全局平均池化+全连接

    def forward(self, img):
        x = img - 0.5
        extract_feature = self.subNetwork1(x)
        attention_map = self.subNetwork2(x)
        intermediate_feature = extract_feature * attention_map
        out = self.subNetwork3(intermediate_feature)
        return out, attention_map


class Discriminator(nn.Module):
    def __init__(self, ndf=64, nChannels=3):
        super(Discriminator, self).__init__()
        # input : (batch * nChannels * image width * image height)
        # Discriminator will be consisted with a series of convolution networks

        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=nChannels,
                out_channels=ndf,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=ndf,
                out_channels=ndf * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=ndf * 2,
                out_channels=ndf * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(
                in_channels=ndf * 4,
                out_channels=ndf * 8,
                kernel_size=4,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(
                in_channels=ndf * 8,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=1,
            ),
            # nn.Sigmoid() -- Replaced with Least Square Loss
        )

    def forward(self, image):
        x = image - .5
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out


class SubNetwork1(nn.Module):
    def __init__(self, ndf=64, ):
        super(SubNetwork1, self).__init__()
        self.ndf = ndf
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, self.ndf, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(self.ndf, self.ndf, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(self.ndf, self.ndf, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf),
            nn.LeakyReLU(inplace=True),
        )
        self.residual_block1 = ResidualBlock(self.ndf)
        self.residual_block2 = ResidualBlock(self.ndf)

    def forward(self, x):
        out = self.conv_block(x)
        out = self.residual_block1(out)
        out = self.residual_block2(out)
        return out


class SubNetwork2(nn.Module):
    def __init__(self, ndf=64, ):
        super(SubNetwork2, self).__init__()
        self.ndf = ndf
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, self.ndf, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf),
            nn.LeakyReLU(inplace=True),
        )
        self.residual_block1 = ResidualBlock(self.ndf)
        self.residual_block2 = ResidualBlock(self.ndf)
        self.residual_block3 = ResidualBlock(self.ndf)
        self.residual_block4 = ResidualBlock(self.ndf)
        self.residual_block5 = ResidualBlock(self.ndf)
        self.last_layer = nn.Sequential(
            nn.Conv2d(self.ndf, 1, kernel_size=3, padding=1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        out = self.conv_block(x)
        out = self.residual_block1(out)
        out = self.residual_block2(out)
        out = self.residual_block3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.last_layer(out)
        result = []
        for item in out:
            item = (item - item.min()) / (item.max() - item.min())
            result.append(item)
        result = torch.stack(result, dim=0)
        # result = (out - out.min()) / (out.max() - out.min())
        # result = torch.sigmoid(out)
        # print(result)
        return result.expand(-1, self.ndf, -1, -1)


class SubNetwork3(nn.Module):
    def __init__(self, ndf=64, message_length=30):
        super(SubNetwork3, self).__init__()
        self.ndf = ndf
        self.message_length = message_length

        self.conv_block = nn.Sequential(
            nn.Conv2d(self.ndf, self.ndf, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf),
            nn.LeakyReLU(inplace=True),
        )
        self.residual_block1 = ResidualBlock(self.ndf)
        self.residual_block2 = ResidualBlock(self.ndf)
        self.residual_block3 = ResidualBlock(self.ndf)
        self.residual_block4 = ResidualBlock(self.ndf)
        self.residual_block5 = ResidualBlock(self.ndf)
        self.residual_block6 = ResidualBlock(self.ndf)
        self.residual_block7 = ResidualBlock(self.ndf)

        self.last_layer = nn.Sequential(
            nn.Conv2d(self.ndf, self.message_length, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.message_length),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(self.message_length, self.message_length * 2, ),
            # nn.BatchNorm1d(self.message_length * 2),
            nn.Identity(),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.message_length * 2, self.message_length),
            # nn.Sigmoid()
        )

    def forward(self, x):
        out = self.conv_block(x)
        out = self.residual_block1(out)
        out = self.residual_block2(out)
        out = self.residual_block3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        out = self.residual_block7(out)
        out = self.last_layer(out)
        return torch.sigmoid(out)
        # return out


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),  # 上上下左右均填充1
                      nn.Conv2d(in_features, in_features, 3, bias=False),
                      nn.BatchNorm2d(in_features),
                      nn.LeakyReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3, bias=False),
                      nn.BatchNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.conv_block(x))


class MaskRCNN(nn.Module):
    def __init__(self, threshold=0.75):
        super(MaskRCNN, self).__init__()
        self.threshold = threshold
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.random_colour_masks = self.random_colour_masks
        for param in self.model.parameters():
            param.requires_grad = False

    # def forward(self, img):
    #     device = img.device
    #
    #     output = self.model(img)
    #     result_mask = []
    #     isNone = False
    #     for num, each_mask in enumerate(output):
    #         pred_score = list(each_mask['scores'].cpu().detach().numpy())
    #         pred_t = [pred_score.index(x) for x in pred_score if x > self.threshold]
    #         if len(pred_t) == 0:
    #             isNone = True
    #             break
    #         pred_t = pred_t[-1]
    #         masks = (each_mask['masks'] > 0.5).squeeze(1).detach().cpu().numpy()
    #         masks = masks[:pred_t + 1]
    #         masks = torch.tensor(masks)
    #         _, idx = torch.sum(masks, dim=[1, 2]).max(0)  # 最大语义区域max,最小语义区域min
    #         idx = idx.item()
    #         rgb_mask = self.random_colour_masks(masks[0])
    #         print(rgb_mask.shape)
    #         mask = cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2GRAY)
    #         _, mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY, cv2.THRESH_OTSU)
    #         out1 = torch.tensor(np.stack([mask, mask, mask], axis=0)).to(device)
    #         out = out1 * 1.0
    #         # rgb_mask = self.random_colour_masks(masks[1])
    #         # print(rgb_mask.shape)
    #         # mask = cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2GRAY)
    #         # _, mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY, cv2.THRESH_OTSU)
    #         # out2 = torch.tensor(np.stack([mask, mask, mask], axis=0)).to(device)
    #         # out = out2 * 1.0
    #         # out=out1 + out2
    #         result_mask.append(out)
    #     if isNone:
    #         return None
    #     result = torch.stack(result_mask, dim=0)
    #     result = result.to(device)
    #     return result


    ##专属马12
    # def forward(self, img):
    #     device = img.device
    #
    #     output = self.model(img)
    #     result_mask = []
    #     isNone = False
    #     for num, each_mask in enumerate(output):
    #         pred_score = list(each_mask['scores'].cpu().detach().numpy())
    #         pred_t = [pred_score.index(x) for x in pred_score if x > self.threshold]
    #         if len(pred_t) == 0:
    #             isNone = True
    #             break
    #         pred_t = pred_t[-1]
    #         masks = (each_mask['masks'] > 0.5).squeeze(1).detach().cpu().numpy()
    #         masks = masks[:pred_t + 1]
    #         masks = torch.tensor(masks)
    #         _, idx = torch.sum(masks, dim=[1, 2]).max(0)  # 最大语义区域max,最小语义区域min
    #         idx = idx.item()
    #         rgb_mask = self.random_colour_masks(masks[0])
    #         # print(rgb_mask.shape)
    #         mask = cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2GRAY)
    #         _, mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY, cv2.THRESH_OTSU)
    #         out1 = torch.tensor(np.stack([mask, mask, mask], axis=0)).to(device)
    #         out1 = out1 * 1.0
    #         rgb_mask = self.random_colour_masks(masks[2])
    #         # print(rgb_mask.shape)
    #         mask = cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2GRAY)
    #         _, mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY, cv2.THRESH_OTSU)
    #         out2 = torch.tensor(np.stack([mask, mask, mask], axis=0)).to(device)
    #         out2 = out2 * 1.0
    #         out=out1 + out2
    #         result_mask.append(out)
    #     if isNone:
    #         return None
    #     result = torch.stack(result_mask, dim=0)
    #     result = result.to(device)
    #     return result



    def forward(self, img):
        device = img.device

        output = self.model(img)
        result_mask = []
        isNone = False
        for num, each_mask in enumerate(output):
            pred_score = list(each_mask['scores'].cpu().detach().numpy())
            pred_t = [pred_score.index(x) for x in pred_score if x > self.threshold]
            if len(pred_t) == 0:
                isNone = True
                break
            pred_t = pred_t[-1]
            masks = (each_mask['masks'] > 0.5).squeeze(1).detach().cpu().numpy()
            masks = masks[:pred_t + 1]
            # print(masks)
            masks = torch.tensor(masks)
            print(masks.size())
            ans, idx = torch.sum(masks, dim=[1, 2]).min(0)  # 最大语义区域max,最小语义区域min
            yuzhi = masks.size()[1] * masks.size()[2] / 8
            # print(yuzhi)
            idx = idx.item()
            visit=[idx]
            print("初始最小区域面积为：%d"%ans)
            if ans >= yuzhi:
                print("最小实例面积已经达到水印嵌入阈值，无需修改,开始下一个区域的操作")
            else:
                print("最小实例面积小于水印嵌入阈值,当前最小实例是实例%d"%idx)
                one_ans,i,flag=near_border(masks,idx,yuzhi,visit)
                print("合并接壤实例区域后的区域面积:%d"%one_ans)
                    # masks[j,:,:]=0
                if flag == 1:
                    print("当前实例区域面积已经大于等于水印嵌入区域面积......开始下一个区域操作")
                    for j in i:
                        masks[idx, :, :] = masks[idx, :, :] + masks[j, :, :]
                        visit.append(j)
                else:
                    if len(i)!=0:
                        for j in i:
                            masks[idx, :, :] = masks[idx, :, :] + masks[j, :, :]
                            visit.append(j)
                    print("当前实例区域面积仍无法满足水印嵌入区域,变成长方形")
                    leftidx, rightidx, upidx, downidx = find_border(masks[idx])
                    print(leftidx,rightidx,downidx,upidx)
                    while one_ans < yuzhi:
                        one_ans = (rightidx - leftidx + 1) * (downidx - upidx + 1)
                        if one_ans < yuzhi:
                            if rightidx < masks.size()[2]:
                                rightidx += 1
                            if leftidx > 0:
                                leftidx -= 1
                            if downidx < masks.size()[2]:
                                downidx += 1
                            if upidx > 0:
                                upidx -= 1
                            # rightidx += 1
                            # leftidx -= 1
                            # downidx += 1
                            # upidx -= 1
                        else:
                            for ii in range(leftidx, rightidx + 1):
                                for jj in range(upidx, downidx + 1):
                                    masks[idx, jj, ii] = True
                            print(leftidx,rightidx)
                            print("当前长方形框面积为%d,已经达到水印区域阈值" % one_ans)
                            print(upidx)
            for i in range(0,masks.size()[0]):
                if i != idx and (i not in visit):
                    visit.append(i)
                    if torch.sum(masks, dim=[1, 2])[i] >= yuzhi:
                        print("区域%d已经达到水印嵌入阈值，无需修改,开始下一个区域的操作"%i)
                    else:
                        print("区域%d面积小于水印嵌入阈值" % i)
                        one_ans, k, flag = near_border(masks, i, yuzhi,visit)
                        # print(k)
                        print("合并接壤实例区域后的区域面积:%d" % one_ans)
                        if flag == 1:
                            print("当前实例区域面积已经大于等于水印嵌入区域面积......开始下一个区域操作")
                        else:
                            if len(k) != 0:
                                for j in k:
                                    if j not in visit:
                                        masks[i, :, :] = masks[i, :, :] + masks[j, :, :]
                                        visit.append(j)
                            print("当前实例区域面积在加入接壤区域后大小仍无法满足水印嵌入区域,变成长方形")
                            leftidx, rightidx, upidx, downidx = find_border(masks[i])
                            while one_ans < yuzhi:
                                one_ans = (rightidx - leftidx + 1) * (downidx - upidx + 1)
                                if one_ans < yuzhi:
                                    if rightidx < masks.size()[2]:
                                        rightidx += 1
                                    if leftidx > 0:
                                        leftidx -= 1
                                    if downidx < masks.size()[2]:
                                        downidx += 1
                                    if upidx > 0:
                                        upidx -= 1
                                else:
                                    for ii in range(leftidx, rightidx + 1):
                                        for jj in range(upidx, downidx + 1):
                                            masks[i, jj, ii] = True
                                    print("当前长方形框面积为%d,已经达到水印区域阈值" % one_ans)
        maskstotal = torch.zeros_like(masks)
        for i in range(0,masks.size()[0]):
            maskstotal[0]=masks[i]+maskstotal[0]
        rgb_mask = self.random_colour_masks(maskstotal[0])
        mask = cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY, cv2.THRESH_OTSU)
        out = torch.tensor(np.stack([mask, mask, mask], axis=0)).to(device)
        out = out * 1.0
        result_mask.append(out)
        if isNone:
            return None
        result = torch.stack(result_mask, dim=0)
        result = result.to(device)
        return result
    @staticmethod
    def random_colour_masks(image):
        colours = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255], [80, 70, 180],
                   [250, 80, 190], [245, 145, 50],
                   [70, 150, 250], [50, 190, 190]]
        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)
        r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0, 10)]
        coloured_mask = np.stack([r, g, b], axis=2)
        return coloured_mask


class DeepHomographyModel(nn.Module):
    def __init__(self, p_nums=4):
        super(DeepHomographyModel, self).__init__()
        self.conv_layer1 = nn.Conv2d(2, 64, 3, padding=1)
        self.conv_layer2 = nn.Conv2d(64, 64, 3, padding=1, stride=2)
        self.conv_layer3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv_layer4 = nn.Conv2d(64, 64, 3, padding=1, stride=2)
        self.conv_layer5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv_layer6 = nn.Conv2d(128, 128, 3, padding=1, stride=2)
        self.conv_layer7 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv_layer8 = nn.Conv2d(128, 128, 3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.batch_norm4 = nn.BatchNorm2d(64)
        self.batch_norm5 = nn.BatchNorm2d(128)
        self.batch_norm6 = nn.BatchNorm2d(128)
        self.batch_norm7 = nn.BatchNorm2d(128)
        self.batch_norm8 = nn.BatchNorm2d(128)
        self.flatten = nn.Flatten()
        self.fc_layer1 = nn.Linear(128 * 26 * 26, 1024)
        self.fc_layer2 = nn.Linear(1024, 2 * p_nums)

    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.batch_norm1(out)
        out = F.relu(out)

        out = self.conv_layer2(out)
        out = self.batch_norm2(out)
        out = F.relu(out)

        out = self.conv_layer3(out)
        out = self.batch_norm3(out)
        out = F.relu(out)

        out = self.conv_layer4(out)
        out = self.batch_norm4(out)
        out = F.relu(out)

        out = self.conv_layer5(out)
        out = self.batch_norm5(out)
        out = F.relu(out)

        out = self.conv_layer6(out)
        out = self.batch_norm6(out)
        out = F.relu(out)

        out = self.conv_layer7(out)
        out = self.batch_norm7(out)
        out = F.relu(out)

        out = self.conv_layer8(out)
        out = self.batch_norm8(out)
        out = F.relu(out)
        # out = out.view(-1, 128 * 16 * 16)
        # print(out.shape)
        out = self.flatten(out)
        out = self.fc_layer1(out)
        out = F.relu(out)
        out = self.fc_layer2(out)
        return out


class DeepModel(nn.Module):
    def __init__(self):
        super(DeepModel, self).__init__()
        self.conv_layer1 = nn.Conv2d(2, 64, 3, padding=1)
        self.conv_layer2 = nn.Conv2d(64, 64, 3, padding=1, stride=2)
        self.conv_layer3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv_layer4 = nn.Conv2d(64, 64, 3, padding=1, stride=2)
        self.conv_layer5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv_layer6 = nn.Conv2d(128, 128, 3, padding=1, stride=2)
        self.conv_layer7 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv_layer8 = nn.Conv2d(128, 128, 3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.batch_norm4 = nn.BatchNorm2d(64)
        self.batch_norm5 = nn.BatchNorm2d(128)
        self.batch_norm6 = nn.BatchNorm2d(128)
        self.batch_norm7 = nn.BatchNorm2d(128)
        self.batch_norm8 = nn.BatchNorm2d(128)
        self.flatten = nn.Flatten()
        self.fc_layer1 = nn.Linear(128 * 26 * 26, 1024)
        self.fc_layer2 = nn.Linear(1024, 8)

    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.batch_norm1(out)
        out = F.relu(out)

        out = self.conv_layer2(out)
        out = self.batch_norm2(out)
        out = F.relu(out)

        out = self.conv_layer3(out)
        out = self.batch_norm3(out)
        out = F.relu(out)

        out = self.conv_layer4(out)
        out = self.batch_norm4(out)
        out = F.relu(out)

        out = self.conv_layer5(out)
        out = self.batch_norm5(out)
        out = F.relu(out)

        out = self.conv_layer6(out)
        out = self.batch_norm6(out)
        out = F.relu(out)

        out = self.conv_layer7(out)
        out = self.batch_norm7(out)
        out = F.relu(out)

        out = self.conv_layer8(out)
        out = self.batch_norm8(out)
        out = F.relu(out)
        # out = out.view(-1, 128 * 16 * 16)
        # print(out.shape)
        out = self.flatten(out)
        out = self.fc_layer1(out)
        out = F.relu(out)
        out = self.fc_layer2(out)
        return out


class DModel(nn.Module):
    def __init__(self):
        super(DModel, self).__init__()
        self.conv_layer1 = nn.Conv2d(6, 64, 3, padding=1)
        self.conv_layer2 = nn.Conv2d(64, 64, 3, padding=1, stride=2)
        self.conv_layer3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv_layer4 = nn.Conv2d(64, 64, 3, padding=1, stride=2)
        self.conv_layer5 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.conv_layer6 = nn.Conv2d(128, 128, 3, padding=1, stride=2)
        self.conv_layer7 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv_layer8 = nn.Conv2d(128, 128, 3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.batch_norm4 = nn.BatchNorm2d(64)
        self.batch_norm5 = nn.BatchNorm2d(128)
        self.batch_norm6 = nn.BatchNorm2d(128)
        self.batch_norm7 = nn.BatchNorm2d(128)
        self.batch_norm8 = nn.BatchNorm2d(128)
        self.flatten = nn.Flatten()
        self.fc_layer1 = nn.Linear(128 * 25 * 25, 1024)
        self.fc_layer2 = nn.Linear(1024, 16)

    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.batch_norm1(out)
        out = F.relu(out)

        out = self.conv_layer2(out)
        out = self.batch_norm2(out)
        out = F.relu(out)

        out = self.conv_layer3(out)
        out = self.batch_norm3(out)
        out = F.relu(out)

        out = self.conv_layer4(out)
        out = self.batch_norm4(out)
        out = F.relu(out)

        out = self.conv_layer5(out)
        out = self.batch_norm5(out)
        out = F.relu(out)

        out = self.conv_layer6(out)
        out = self.batch_norm6(out)
        out = F.relu(out)

        out = self.conv_layer7(out)
        out = self.batch_norm7(out)
        out = F.relu(out)

        out = self.conv_layer8(out)
        out = self.batch_norm8(out)
        out = F.relu(out)
        # out = out.view(-1, 128 * 16 * 16)
        # print(out.shape)
        out = self.flatten(out)
        out = self.fc_layer1(out)
        out = F.relu(out)
        out = self.fc_layer2(out)
        return out


if __name__ == "__main__":
    encoder = Encoder()
    decoder = Decoder()
    discriminator = Discriminator()
    pos_net = DeepHomographyModel()
    premodel = DModel()
    dm = DeepModel()

    # image = torch.rand(1, 3, 256, 256)
    # sec = torch.ones(1, 30)
    # extract = torch.randn(10, 3, 256, 256)
    # enc = encoder(image, sec)
    # dec = decoder(extract)
    # dis = discriminator(image)
    inputs_pots = torch.randn(1, 6, 400, 400)
    pos = premodel(inputs_pots)
    # print(pos.shape)

    # pos = torch.randn(1, 1, 129, 129)
    # p = premodel(pos)
    # print(p.shape)
    # print_network(decoder)
    # stn_out = stn(image)
    # print(stn_out.shape)
    # print(enc.shape, dec.shape, dis.shape)
    # print_network(encoder)
    # print(encoder)
    # print(sec.shape)
    # dis = Discriminator()
    # out = dis(image)
