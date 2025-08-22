# !/usr/bin/env python
# -- coding: utf-8 --
# 小何一定行
import os
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
import torch


class myDataset(Dataset):
    def __init__(self, args, isTrain=True):
        super(myDataset, self).__init__()
        self.img_size = (args.image_size, args.image_size)
        self.message_length = args.message_length
        if isTrain:
            self.img_path = args.datarootTrain
            self.transform = transforms.Compose([transforms.Resize(self.img_size, Image.BICUBIC),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(), ])
        else:
            self.img_path = args.datarootTest
            self.transform = transforms.Compose([transforms.Resize(self.img_size, Image.BICUBIC),
                                                 transforms.ToTensor(), ])

        self.files_img = sorted([x for x in os.listdir(self.img_path)])

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_path, self.files_img[idx % len(self.files_img)])
        item_img = self.transform(Image.open(img_path).convert('RGB'))  # RGB载体图

        item_secret = np.random.binomial(1, 0.5, self.message_length)  # 秘密消息
        item_secret = torch.from_numpy(item_secret).float()

        return {'img': item_img, 'sec': item_secret}

    def __len__(self):
        return len(self.files_img)
