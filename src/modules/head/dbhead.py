from typing import *
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class DBHead(nn.Module):
    def __init__(self, in_chans, k=50,
                 bias=False, *args, **kwargs):
        """
            adaptive: adaptive threshold 사용여부
            smooth: threshold upsample과정에서 interpolate를 사용할지 deconv를 사용할지 여부
            serial: threshold 예측할 때, probabilty맵을 추가적으로 concat하여 사용할지 여부
        """

        super(DBHead, self).__init__()

        self.k = k
        self.binarize = nn.Sequential(
            nn.Conv2d(in_chans, in_chans//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_chans//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_chans//4, in_chans//4, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_chans//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_chans//4, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )


        self.binarize.apply(self.weights_init)
        self.threshold = self._init_threshold(in_chans, bias=bias)
        self.threshold.apply(self.weights_init)


    def forward(self, fuse_inp):
        binary = self.binarize(fuse_inp)

        if self.training:
            threshold = self.threshold(fuse_inp)
            approxi_binary = self.calc_approx_binary(binary, threshold)
            res = torch.cat((binary, threshold, approxi_binary), dim=1)
        else:
            res = binary
        return res


    def calc_approx_binary(self, binary, threshold):
        return torch.reciprocal(1 + torch.exp(-self.k * (binary - threshold)))


    def _init_upsample(self, in_chans, out_chans, bias=False):
        return nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)


    def _init_threshold(self, in_chans, bias=False):
        threshold = nn.Sequential(
            nn.Conv2d(in_chans, in_chans//4, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(in_chans//4),
            nn.ReLU(inplace=True),
            self._init_upsample(in_chans//4, in_chans//4, bias=bias),
            nn.BatchNorm2d(in_chans//4),
            nn.ReLU(inplace=True),
            self._init_upsample(in_chans//4, out_chans=1, bias=bias),
            nn.Sigmoid()
        )

        return threshold


    def weights_init(self, m):
        classname = m.__class__.__name__

        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)