import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import random
import numpy as np

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w

    new_h = new_w = opt.loadSize
    if 'crop' in opt.resize_or_crop:
        x = random.randint(0, np.maximum(0, new_w - opt.fineSize))
        y = random.randint(0, np.maximum(0, new_h - opt.fineSize))

    if not opt.no_flip:
        flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}

def get_transform(opt, params=None):
    transform_list = []
    if 'resize' in opt.resize_or_crop:
        transform_list.append(transforms.Resize(opt.loadSize, Image.BICUBIC))

    if opt.isTrain:
        
        if 'crop' in opt.resize_or_crop:
            if params is None: transform_list.append(transforms.RandomCrop(opt.fineSize))
            else: transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.fineSize)))
        if not opt.no_flip:
            if params is None: transform_list.append(transforms.RandomHorizontalFlip())
            elif params['flip']: transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))
                

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img