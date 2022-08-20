import math

import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair


class DepthAdaptScaleConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 deformable_groups=1,
                 bias=False):
        super(DepthAdaptScaleConv, self).__init__()

        assert not bias
        assert in_channels % groups == 0, \
            'in_channels {} cannot be divisible by groups {}'.format(
                in_channels, groups)
        assert out_channels % groups == 0, \
            'out_channels {} cannot be divisible by groups {}'.format(
                out_channels, groups)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // self.groups,
                         *self.kernel_size))

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
    
    def adaptive_conv(self, scale_coeff, x):
        assert scale_coeff.shape[0] == x.shape[2]
        
        pass

    def forward(self, x, offset):
        return deform_conv(x, offset, self.weight, self.stride, self.padding,
                           self.dilation, self.groups, self.deformable_groups)


class DepthAdaptScaleConvPack(DepthAdaptScaleConv):

    def __init__(self, *args, **kwargs):
        super(DeformConvPack, self).__init__(*args, **kwargs)

        self.conv_scale_coeff = nn.Conv2d(self.in_channels * 2, 1,
            kernel_size=self.kernel_size,
            stride=(1,1),
            padding=_pair(self.padding),
            bias=True)
        self.init_coeff()

    def init_coeff(self):
        self.conv_scale_coeff.weight.data.zero_()
        self.conv_scale_coeff.bias.data.zero_()

    def forward(self, x):
        offset = self.conv_scale_coeff(x)
        return deform_conv(x, offset, self.weight, self.stride, self.padding,
                           self.dilation, self.groups, self.deformable_groups)