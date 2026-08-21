import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

class Generator(nn.Module):
    """
    生成器
    """
    def __init__(self):
        super(Generator, self).__init__()

        # 全连接层
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]  # 线性层
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))  # 批量归一化
            layers.append(nn.LeakyReLU(0.2, inplace=True))  # 激活函数
            return layers

        # 生成器网络
        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()  # 归一化，[-1,1]
        )

    # 向前传播
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    """
    判别器
    """
    def __init__(self):
        super(Discriminator, self).__init__()

        # 判别器网络
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


