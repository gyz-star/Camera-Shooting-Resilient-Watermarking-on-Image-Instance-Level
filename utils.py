import string
import os
os.environ['CUDA_VISIBLE_DEVICE']='1'
import cv2
import numpy as np
import random
import torch
import torchgeometry
from torch.nn import init
from torch import autograd, nn
import torch.nn.functional as F
import kornia as K
from torchvision import transforms


def weights_init(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def find_border(matr):
    for i in range(0,matr.size()[0]-1):
        if torch.sum(matr,dim=[0])[i]==False and torch.sum(matr,dim=[0])[i+1]!=False:
            leftidx=i
        if torch.sum(matr,dim=[0])[i]!=False and torch.sum(matr,dim=[0])[i+1]==False:
            rightidx=i
    for j in range(0, matr.size()[1]-1):
        if torch.sum(matr, dim=[1])[j] == False and torch.sum(matr, dim=[1])[j + 1] != False:
            upidx=j
        if torch.sum(matr, dim=[1])[j] != False and torch.sum(matr, dim=[1])[j + 1] == False:
            downidx=j
    return leftidx,rightidx,upidx,downidx
def near_border(masks,idx,yuzhi,visit):
    masksleft = torch.zeros_like(masks)
    masksright = torch.zeros_like(masks)
    masksdown = torch.zeros_like(masks)
    masksup = torch.zeros_like(masks)
    masksumup = torch.zeros_like(masks)
    masksumdown = torch.zeros_like(masks)
    masksumleft = torch.zeros_like(masks)
    masksumright = torch.zeros_like(masks)
    ans=torch.sum(masks, dim=[1, 2])[idx]
    nearidx=[]
    flag=0
    for i in range(0, masks.size()[0]):
        if i != idx and i not in visit:
            # print(idx)
            # print(i)
            print("正在判断实例%d是否与最小实例接壤......" % i)
            for j in range(0, masks.size()[1] - 1):
                for k in range(0, masks.size()[2] - 2):
                    masksleft[i, j, k] = masks[i, j, k + 1]
                masksleft[i, j, 255] = False
            for j in range(0, masks.size()[1] - 1):
                for k in range(1, masks.size()[2] - 1):
                    masksright[i, j, k] = masks[i, j, k - 1]
                masksright[i, j, 0] = False
            for k in range(0, masks.size()[1] - 1):
                for j in range(0, masks.size()[2] - 2):
                    masksup[i, j, k] = masks[i, j + 1, k]
                masksup[i, 255, k] = False
            for k in range(0, masks.size()[1] - 1):
                for j in range(1, masks.size()[2] - 1):
                    masksdown[i, j, k] = masks[i, j - 1, k]
                masksdown[i, 0, k] = False
            for k in range(0, 255):
                for j in range(0, 255):
                    masksumleft[i, k, j] = (masksleft[i, k, j] + masks[idx, k, j]).long() % 2
                    masksumright[i, k, j] = (masksright[i, k, j] + masks[idx, k, j]).long() % 2
                    masksumdown[i, k, j] = (masksdown[i, k, j] + masks[idx, k, j]).long() % 2
                    masksumup[i, k, j] = (masksup[i, k, j] + masks[idx, k, j]).long() % 2
            if torch.sum(masksumleft, dim=[1, 2])[i] < torch.sum(masks, dim=[1, 2])[idx] + torch.sum(masks, dim=[1, 2])[i] or torch.sum(masksumright, dim=[1, 2])[i] < torch.sum(masks, dim=[1, 2])[idx] +torch.sum(masks, dim=[1, 2])[i] or torch.sum(masksumdown, dim=[1, 2])[i] <torch.sum(masks, dim=[1, 2])[idx] + torch.sum(masks, dim=[1, 2])[i] or torch.sum(masksumup, dim=[1, 2])[i] < torch.sum(masks, dim=[1, 2])[idx] +torch.sum(masks, dim=[1, 2])[i]:
                ans=ans+torch.sum(masks, dim=[1, 2])[i]
                nearidx.append(i)
                # masks[i,:,:]=False
                if ans >= yuzhi:
                    flag=1
                    return ans,nearidx,flag
    return ans,nearidx,flag


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = True  # False
    torch.backends.cudnn.deterministic = True


def get_secret_acc(secret_pred, secret_true):
    # 比特精确度,和秘密消息正确率
    if 'cuda' in str(secret_pred.device):
        secret_pred = secret_pred.cpu()
        secret_true = secret_true.cpu()
    secret_pred = torch.abs(torch.round(secret_pred).clip(0, 1))
    correct_pred = torch.sum((secret_pred - secret_true) == 0, dim=1)
    str_acc = 1.0 - torch.sum((correct_pred - secret_pred.size()[1]) != 0).numpy() / correct_pred.size()[0]
    bit_acc = torch.sum(correct_pred).numpy() / secret_pred.numel()  # numel()函数：返回数组中元素的个数
    return bit_acc


def get_rand_homography_mat(img, eps=0.2):
    batch_size = img.shape[0]
    res = np.zeros((batch_size, 2, 3, 3))

    h, w = img.shape[2:4]
    eps_x = w * eps
    eps_y = h * eps
    for i in range(batch_size):
        top_left_x = random.uniform(-eps_x, eps_x)
        top_left_y = random.uniform(-eps_y, eps_y)
        bottom_left_x = random.uniform(-eps_x, eps_x)
        bottom_left_y = random.uniform(-eps_y, eps_y)
        top_right_x = random.uniform(-eps_x, eps_x)
        top_right_y = random.uniform(-eps_y, eps_y)
        bottom_right_x = random.uniform(-eps_x, eps_x)
        bottom_right_y = random.uniform(-eps_y, eps_y)

        rect = np.array([
            [top_left_x, top_left_y],
            [top_right_x + w, top_right_y],
            [bottom_right_x + w, bottom_right_y + h],
            [bottom_left_x, bottom_left_y + h]], dtype="float32")

        dst = np.array([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]], dtype="float32")

        res_i = cv2.getPerspectiveTransform(rect, dst)
        res_i_inv = np.linalg.inv(res_i)

        res[i, 0, :, :] = res_i
        res[i, 1, :, :] = res_i_inv
    return res


def get_warped_points_and_gap(source_points: np.ndarray, homography: torch.tensor, ):
    device = homography.device
    nums = len(source_points)
    source_points_ = source_points.reshape((1, nums, 2))
    warped_points = []
    gaps = []
    for i in range(len(homography)):
        H = homography[i, 0].cpu().numpy()
        H_inverse = homography[i, 1].cpu().numpy()
        warped_point = cv2.perspectiveTransform(np.float32(source_points_), H_inverse).reshape(-1, 2)
        gap = torch.from_numpy(np.subtract(np.array(warped_point), np.array(source_points)))
        gaps.append(gap)
        warped_points.append(warped_point)
    targets = torch.stack(gaps, dim=0).float().to(device)
    points = np.stack(warped_points, axis=0)
    return targets, points


def get_min_max_scale_extract(batch: torch.tensor):
    res = []
    device = batch.device
    for i in range(len(batch)):
        temp = (batch[i] - torch.min(batch[i])) / (torch.max(batch[i]) - torch.min(batch[i]))
        res.append(temp)
    return torch.stack(res, dim=0).to(device)


def get_warped_extracted_IMG(images, masks, args, isMul=True, isHomography=True, isResize=True):
    device = images.device
    if isHomography:
        h, w = images.shape[2:4]
        homography = get_rand_homography_mat(images)
        homography = torch.from_numpy(homography).float().to(device)

        transformDec = torchgeometry.warp_perspective(images, homography[:, 1, :, :], dsize=(h, w))
        transformDec = torchgeometry.warp_perspective(transformDec, homography[:, 0, :, :], dsize=(h, w))
        transformMask = torchgeometry.warp_perspective(masks, homography[:, 1, :, :], dsize=(h, w))
        transformMask = torchgeometry.warp_perspective(transformMask, homography[:, 0, :, :], dsize=(h, w))
        transformMask[transformMask > 0] = 1.0
    else:
        transformDec, transformMask = images, masks

    waitDec = []
    for img, mask in zip(transformDec, transformMask):
        if len(torch.unique(mask)) == 1 and torch.unique(mask)[0] == 0:
            return None
        index = torch.where(mask == 1)
        h_min, h_max = torch.min(index[1]), torch.max(index[1])
        w_min, w_max = torch.min(index[2]), torch.max(index[2])

        partialIMG = img[:, h_min:h_max, w_min:w_max].clone()
        partialMask = mask[:, h_min:h_max, w_min:w_max].clone()
        partialIMG = partialIMG.unsqueeze(0)
        partialMask = partialMask.unsqueeze(0)

        assert partialIMG.shape == partialMask.shape

        if isMul:
            dec = partialIMG * partialMask
        else:
            dec = partialIMG
        if isResize:
            dec = F.interpolate(dec, size=(args.extract_size, args.extract_size), )
        waitDec.append(dec)

    return torch.cat(waitDec, dim=0)


def extractIMG(images, masks, args, isMul=True, isHomography=True, isResize=True, isBack=True):
    waitDec = []
    device = images.device

    for img, mask in zip(images, masks):
        if len(torch.unique(mask)) == 1 and torch.unique(mask)[0] == 0:
            return None
        index = torch.where(mask == 1)
        h_min, h_max = torch.min(index[1]), torch.max(index[1])
        w_min, w_max = torch.min(index[2]), torch.max(index[2])

        partialIMG = img[:, h_min:h_max, w_min:w_max].clone()
        partialMask = mask[:, h_min:h_max, w_min:w_max].clone()
        partialIMG = partialIMG.unsqueeze(0)
        partialMask = partialMask.unsqueeze(0)

        assert partialIMG.shape == partialMask.shape
        h, w = partialIMG.shape[2:4]
        # print(h, w)
        if isHomography:
            homography = get_rand_homography_mat(partialIMG)
            homography = torch.from_numpy(homography).float().to(device)
            transformDec = torchgeometry.warp_perspective(partialIMG, homography[:, 1, :, :], dsize=(h, w))
            transformMask = torchgeometry.warp_perspective(partialMask, homography[:, 1, :, :], dsize=(h, w))
            if isBack:
                transformDec = torchgeometry.warp_perspective(transformDec, homography[:, 0, :, :], dsize=(h, w))
                transformMask = torchgeometry.warp_perspective(transformMask, homography[:, 0, :, :], dsize=(h, w))
                transformMask[transformMask > 0] = 1.0
        else:
            transformDec, transformMask = partialIMG, partialMask

        if isMul:
            dec = transformDec * transformMask
        else:
            dec = transformDec
        if isResize:
            dec = F.interpolate(dec, size=(args.extract_size, args.extract_size))
        waitDec.append(dec)

    return torch.cat(waitDec, dim=0)
    # return waitDec


def extract_img_mask(images, masks, extract_size=256, isMul=False, isResize=True):
    waitDec = []
    waitMask = []
    # waitDec = torch.Tensor([])
    # waitMask = torch.Tensor([])
    device = images.device

    for img, mask in zip(images, masks):
        if len(torch.unique(mask)) == 1 and torch.unique(mask)[0] == 0:
            return None
        index = torch.where(mask == 1)
        h_min, h_max = torch.min(index[1]), torch.max(index[1])
        w_min, w_max = torch.min(index[2]), torch.max(index[2])

        partialIMG = img[:, h_min:h_max, w_min:w_max].clone()
        partialMask = mask[:, h_min:h_max, w_min:w_max].clone()
        partialIMG = partialIMG.unsqueeze(0)
        partialMask = partialMask.unsqueeze(0)

        assert partialIMG.shape == partialMask.shape
        h, w = partialIMG.shape[2:4]
        # print(h, w)

        transformDec, transformMask = partialIMG, partialMask

        if isMul:
            dec = transformDec * transformMask
        else:
            dec = transformDec
        if isResize:
            dec = F.interpolate(dec, size=(extract_size, extract_size))
            mk = F.interpolate(transformMask, size=(extract_size, extract_size))
            mk[mk > 0] = 1.0
            waitMask.append(mk)
            waitDec.append(dec)
            print(dec.shape, mk.shape)

    return torch.cat(waitDec, dim=0).to(device), torch.cat(waitMask, dim=0).to(device)
    # return waitDec


def tensor2im(image_tensor, imtype=np.uint8, normalize=False, is255=False):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.squeeze(0).detach().cpu().float().numpy()
    if not is255:
        if normalize:
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        else:
            image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:
        image_numpy = image_numpy[:, :, 0]
    return image_numpy.astype(imtype)


def im2tensor(img):
    assert type(img) == np.ndarray, 'the img type is {}, but ndarry expected'.format(type(img))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255).unsqueeze(0)  # 255也可以改为256


def min_max_scale(num):
    return (num - np.min(num)) / (np.max(num) - np.min(num))


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def get_cdos(img, dos_num=9):
    h, w = img.shape[2:4]
    cdos = None
    # # 8个点
    # top_left = [temp_w, temp_h]
    # top_mid = [temp_w * 2, temp_h]
    # top_right = [temp_w * 3, temp_h]
    # mid_left = [temp_w, temp_h * 2]
    # mid_right = [temp_w * 3, temp_h * 2]
    # bottom_left = [temp_w, temp_h * 3]
    # bottom_mid = [temp_w * 2, temp_h * 3]
    # bottom_right = [temp_w * 3, temp_h * 3]
    # cdos = np.array([top_left, top_mid, top_right, mid_right, bottom_right, bottom_mid, bottom_left, mid_left])
    if dos_num == 9:
        # 9个点
        ex_h = int(h // 2)
        ex_w = int(w // 2)
        temp_h = max(int(ex_h // 3) - 12, 2)
        temp_w = max(int(ex_w // 3) - 12, 2)
        top_left = [temp_w, temp_h]
        top_mid = [temp_w * 2, temp_h]
        top_right = [temp_w * 3, temp_h]
        mid_left = [temp_w, temp_h * 2]
        mid_mid = [temp_w * 2, temp_h * 2]
        mid_right = [temp_w * 3, temp_h * 2]
        bottom_left = [temp_w, temp_h * 3]
        bottom_mid = [temp_w * 2, temp_h * 3]
        bottom_right = [temp_w * 3, temp_h * 3]
        cdos = np.array([top_left, top_mid, top_right, mid_right, bottom_right, bottom_mid, bottom_left, mid_left, mid_mid])
    elif dos_num == 4:
        # 4个点
        ex_h = int(h // 6) - h // 25
        ex_w = int(w // 6) - w // 25
        temp_h = max(ex_h, 2)
        temp_w = max(ex_w, 2)
        top_left = [temp_w, temp_h]
        top_right = [temp_w * 3, temp_h]
        bottom_left = [temp_w, temp_h * 3]
        bottom_right = [temp_w * 3, temp_h * 3]
        cdos = np.array([top_left, top_right, bottom_right, bottom_left])
    return cdos


def get_cdos_according_mask(images, masks):
    cdos = []
    device = images.device

    for img, mask in zip(images, masks):
        if len(torch.unique(mask)) == 1 and torch.unique(mask)[0] == 0:
            return None
        index = torch.where(mask == 1)
        h_min, h_max = torch.min(index[1]), torch.max(index[1])
        w_min, w_max = torch.min(index[2]), torch.max(index[2])
        h = h_max - h_min
        w = w_max - w_min
        ex_h = int(h // 6.5)
        ex_w = int(w // 6.5)
        temp_h = max(ex_h, 2)
        temp_w = max(ex_w, 2)
        top_left = [w_min // 4 + temp_w, h_min // 4 + temp_h]
        top_mid = [w_min // 4 + temp_w * 2, h_min // 4 + temp_h]
        top_right = [w_min // 4 + temp_w * 3, h_min // 4 + temp_h]
        mid_left = [w_min // 4 + temp_w, h_min // 4 + temp_h * 2]
        mid_mid = [w_min // 4 + temp_w * 2, h_min // 4 + temp_h * 2]
        mid_right = [w_min // 4 + temp_w * 3, h_min // 4 + temp_h * 2]
        bottom_left = [w_min // 4 + temp_w, h_min // 4 + temp_h * 3]
        bottom_mid = [w_min // 4 + temp_w * 2, h_min // 4 + temp_h * 3]
        bottom_right = [w_min // 4 + temp_w * 3, h_min // 4 + temp_h * 3]
        cdo = np.array([top_left, top_mid, top_right, mid_right, bottom_right, bottom_mid, bottom_left, mid_left, mid_mid])
        cdos.append(cdo)
    return cdos


def Random_noise(inputs, noise_factor=0.15):
    device = inputs.device
    noise = torch.rand(inputs.shape) * noise_factor
    noisy = inputs + noise.to(device)
    noisy = torch.clip(noisy, 0., 1.)
    return noisy


def Gauss_noise(img, rnd_noise):
    device = img.device
    # rnd_noise = torch.rand(1)[0] * rnd_noise  # rnd_noise:0.02
    noise = torch.normal(mean=0, std=rnd_noise, size=img.size(), dtype=torch.float32)
    if torch.cuda.is_available():
        noise = noise.to(device)
    img = img + noise  # 实际上也就是原图+噪音
    img = torch.clamp(img, 0, 1)
    return img


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    # brightness=0, contrast=0, saturation=0, hue=0
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([
        rnd_color_jitter,
        rnd_gray])

    return color_distort


# 1. RGB -> YCbCr
class rgb_to_ycbcr_jpeg(nn.Module):
    """ Converts RGB image to YCbCr
    Input:
        image(tensor): batch x 3 x height x width
    Outpput:
        result(tensor): batch x height x width x 3
    """

    def __init__(self):
        super(rgb_to_ycbcr_jpeg, self).__init__()
        matrix = np.array(
            [[0.299, 0.587, 0.114],
             [-0.168736, -0.331264, 0.5],
             [0.5, -0.418688, -0.081312]], dtype=np.float32).T
        self.shift = torch.tensor([0., 128., 128.])  # nn.Parameter(
        self.matrix = torch.from_numpy(matrix)

    def forward(self, x):
        image = (x * 255).permute(0, 2, 3, 1)  # 转为batch*h*w*c的格式
        result = torch.tensordot(image, self.matrix, dims=1) + self.shift
        result = result.reshape(x.shape)
        result = torch.clamp(result, 0, 255)
        return result  # b*h*w*3


class ycbcr_to_rgb_jpeg(nn.Module):
    """ Converts YCbCr image to RGB JPEG
    Input:
        image(tensor): batch x height x width x 3
    Outpput:
        result(tensor): batch x 3 x height x width
    """

    def __init__(self):
        super(ycbcr_to_rgb_jpeg, self).__init__()

        matrix = np.array(
            [[1., 0., 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]],
            dtype=np.float32).T
        self.shift = torch.tensor([0, -128., -128.])  # nn.Parameter
        self.matrix = torch.from_numpy(matrix)

    def forward(self, x):
        image = x.permute(0, 2, 3, 1)  # 转为batch*h*w*c的格式
        result = torch.tensordot(image + self.shift, self.matrix, dims=1)
        result = result.reshape(x.shape)
        result = torch.clamp(result, 0, 255) / 255
        return result  # batch*c*h*w


def convert_rgb_to_ycbcr(x):
    """
            RGB转YCBCR
            Y=0.257*R+0.564*G+0.098*B+16
            Cb=-0.148*R-0.291*G+0.439*B+128
            Cr=0.439*R-0.368*G-0.071*B+128
    """
    # img = x * 255
    img = x
    device = img.device
    if type(img) == torch.Tensor:
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        # y = 16. + (0.257 * img[:, 0, :, :].unsqueeze(1) + 0.564 * img[:, 1, :, :].unsqueeze(1) + 0.098 * img[:, 2, :, :].unsqueeze(1))
        # cb = 128. + (-0.148 * img[:, 0, :, :].unsqueeze(1) - 0.291 * img[:, 1, :, :].unsqueeze(1) + 0.439 * img[:, 2, :, :].unsqueeze(1))
        # cr = 128. + (0.439 * img[:, 0, :, :].unsqueeze(1) - 0.368 * img[:, 1, :, :].unsqueeze(1) - 0.071 * img[:, 2, :, :].unsqueeze(1))

        y = 0.299 * img[:, 0, :, :].unsqueeze(1) + 0.587 * img[:, 1, :, :].unsqueeze(1) + 0.114 * img[:, 2, :, :].unsqueeze(1)
        cr = (img[:, 0, :, :].unsqueeze(1) - y) * 0.713 + 0.5
        cb = (img[:, 2, :, :].unsqueeze(1) - y) * 0.564 + 0.5

        # y = 0.2990 * img[:, 0, :, :].unsqueeze(1) + 0.587 * img[:, 1, :, :].unsqueeze(1) + 0.114 * img[:, 2, :, :].unsqueeze(1)
        # cb = -0.1687 * img[:, 0, :, :].unsqueeze(1) - 0.3313 * img[:, 1, :, :].unsqueeze(1) + 0.5 * img[:, 2, :, :].unsqueeze(1)
        # cr = 0.5 * img[:, 0, :, :].unsqueeze(1) - 0.4187 * img[:, 1, :, :].unsqueeze(1) - 0.0813 * img[:, 2, :, :].unsqueeze(1)
        return torch.cat([y, cb, cr], 1).to(device)  # .permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))


def convert_ycbcr_to_rgb(x):
    """
            YCBCR转RGB
            R=1.164*(Y-16)+1.596*(Cr-128)
            G=1.164*(Y-16)-0.392*(Cb-128)-0.813*(Cr-128)
            B=1.164*(Y-16)+2.017*(Cb-128)
    """
    device = x.device
    # img = x * 255
    img = x
    if type(img) == torch.Tensor:
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        # r = 1.164 * (img[:, 0, :, :].unsqueeze(1) - 16) + 1.596 * (img[:, 2, :, :].unsqueeze(1) - 128)
        # g = 1.164 * (img[:, 0, :, :].unsqueeze(1) - 16) - 0.392 * (img[:, 1, :, :].unsqueeze(1) - 128) - 0.813 * (img[:, 2, :, :].unsqueeze(1) - 128)
        # b = 1.164 * (img[:, 0, :, :].unsqueeze(1) - 16) + 2.017 * (img[:, 1, :, :].unsqueeze(1) - 128)

        b = (img[:, 1, :, :].unsqueeze(1) - 0.5) * 1. / 0.564 + img[:, 0, :, :].unsqueeze(1)
        r = (img[:, 2, :, :].unsqueeze(1) - 0.5) * 1. / 0.713 + img[:, 0, :, :].unsqueeze(1)
        g = 1. / 0.587 * (img[:, 0, :, :].unsqueeze(1) - 0.299 * r - 0.114 * b)

        return (torch.cat([r, g, b], 1)).to(device)
    else:
        raise Exception('Unknown Type', type(img))


class RgbYcbcr(nn.Module):
    def __init__(self):
        super(RgbYcbcr, self).__init__()
        self.Ycbcr = nn.Conv2d(3, 3, 1, 1, bias=False)
        trans_matrix = np.array([[0.299, 0.587, 0.114],
                                 [-0.168736, -0.331264, 0.5],
                                 [0.5, -0.418688, -0.081312]])
        trans_matrix = torch.from_numpy(trans_matrix).float().unsqueeze(2).unsqueeze(3)
        self.Ycbcr.weight.data = trans_matrix
        self.Ycbcr.weight.requires_grad = False

        self.reYcbcr = nn.Conv2d(3, 3, 1, 1, bias=False)
        re_matrix = np.linalg.pinv(np.array([[0.299, 0.587, 0.114],
                                             [-0.168736, -0.331264, 0.5],
                                             [0.5, -0.418688, -0.081312]]))
        re_matrix = torch.from_numpy(re_matrix).float().unsqueeze(2).unsqueeze(3)
        self.reYcbcr.weight.data = re_matrix

    def forward(self, x):
        ycbcr = self.Ycbcr(x)  # b 3 h w
        return ycbcr

    def reverse(self, x):
        rgb = self.reYcbcr(x)
        return rgb


def rgb2ycbcr_tensor(imgs):
    device = imgs.device
    b = imgs.shape[0]
    res = []
    for i in range(b):
        # tensor->ndarray->ycbcr->tensor
        tmp = im2tensor(cv2.cvtColor(tensor2im(imgs[i].clone().unsqueeze(0)), cv2.COLOR_RGB2YCrCb))
        res.append(tmp)
    return torch.cat(res, dim=0).to(device)


def ycbcr2rgb_tensor(imgs):
    device = imgs.device
    b = imgs.shape[0]
    res = []
    for i in range(b):
        # tensor->ndarray->rgb->tensor
        tmp = im2tensor(cv2.cvtColor(tensor2im(imgs[i].clone().unsqueeze(0)), cv2.COLOR_YCrCb2RGB))
        res.append(tmp)
    return torch.cat(res, dim=0).to(device)


def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))


def generate_random_key(l=30):
    s = string.ascii_lowercase + string.digits
    return ''.join(random.sample(s, l))


if __name__ == '__main__':
    img = torch.rand(10, 3, 256, 256)
    cdos = get_cdos(img)

    # rgb_ycbcr = rgb_to_ycbcr_jpeg()
    # ycbcr_rgb = ycbcr_to_rgb_jpeg()
    # print(rgb_ycbcr(img).shape, ycbcr_rgb(img).shape)
    # print(rgb_ycbcr(img).max(), ycbcr_rgb(img).max())

    print(convert_rgb_to_ycbcr(img).shape, convert_rgb_to_ycbcr(img).max())
    print(convert_ycbcr_to_rgb(img).shape, convert_ycbcr_to_rgb(img).max())
    # print(img[0].unsqueeze(0).shape, tensor2im(img[0].unsqueeze(0)).shape)
    # print(cdos.shape)
