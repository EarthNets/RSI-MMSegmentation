# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import swin_t, resnet18, resnet50


class ConvBnLeakyRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.2)


class MiniInception(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MiniInception, self).__init__()
        self.conv1_left  = ConvBnLeakyRelu2d(in_channels,   out_channels//2)
        self.conv1_right = ConvBnLeakyRelu2d(in_channels,   out_channels//2, padding=2, dilation=2)
        self.conv2_left  = ConvBnLeakyRelu2d(out_channels,  out_channels//2)
        self.conv2_right = ConvBnLeakyRelu2d(out_channels,  out_channels//2, padding=2, dilation=2)
        self.conv3_left  = ConvBnLeakyRelu2d(out_channels,  out_channels//2)
        self.conv3_right = ConvBnLeakyRelu2d(out_channels,  out_channels//2, padding=2, dilation=2)
    def forward(self,x):
        x = torch.cat((self.conv1_left(x), self.conv1_right(x)), dim=1)
        x = torch.cat((self.conv2_left(x), self.conv2_right(x)), dim=1)
        x = torch.cat((self.conv3_left(x), self.conv3_right(x)), dim=1)
        return x


class MFNet(nn.Module):

    def __init__(self, n_class):
        super(MFNet, self).__init__()
        rgb_ch = [32,64,128,256]
        inf_ch = [32,64,128,256]

        self.resnet_rgb = resnet18(pretrained=True)
        self.resnet_h = resnet18(pretrained=True)
        #modules_rgb = list(resnet_rgb.children())[:-2]
        #modules_h = list(resnet_h.children())[:-2]
        #self.backbone_rgb = nn.Sequential(*modules_rgb)
        #self.backbone_h = nn.Sequential(*modules_h)
        self.enc_channel = nn.Conv2d(1024,512,1)

        self.decode4     = ConvBnLeakyRelu2d(rgb_ch[3]+inf_ch[3], rgb_ch[2]+inf_ch[2])
        self.decode3     = ConvBnLeakyRelu2d(rgb_ch[2]+inf_ch[2], rgb_ch[1]+inf_ch[1])
        self.decode2     = ConvBnLeakyRelu2d(rgb_ch[1]+inf_ch[1], rgb_ch[0]+inf_ch[0])
        self.decode1     = ConvBnLeakyRelu2d(rgb_ch[0]+inf_ch[0], n_class)

    def forward(self, x):
        # split data into RGB and INF
        x_rgb = x[:,:3]
        x_inf = x[:,3:]
        n,c,h,w = x_inf.shape
        x_h = torch.cat([x_inf,x_inf,x_inf],dim=1)

        # encode
        x_rgbs = self.resnet_rgb(x_rgb)
        x_infs = self.resnet_h(x_h)
        x_rgb = x_rgbs[0]
        x_inf = x_infs[0]
        x_rgb_p4 = x_rgbs[1]
        x_rgb_p3 = x_rgbs[2]
        x_rgb_p2 = x_rgbs[3]

        x_inf_p4 = x_infs[1]
        x_inf_p3 = x_infs[2]
        x_inf_p2 = x_infs[3]

        #print(x_rgb.shape, x_rgb_p4.shape, x_rgb_p3.shape, x_rgb_p2.shape)

        x = torch.cat((x_rgb, x_inf), dim=1) # fusion RGB and INF
        x = self.enc_channel(x)

        # decode
        x = F.upsample(x, scale_factor=2, mode='nearest') # unpool4
        x = self.decode4(x + torch.cat((x_rgb_p4, x_inf_p4), dim=1))
        x = F.upsample(x, scale_factor=2, mode='nearest') # unpool3
        x = self.decode3(x + torch.cat((x_rgb_p3, x_inf_p3), dim=1))
        x = F.upsample(x, scale_factor=2, mode='nearest') # unpool2
        x = self.decode2(x + torch.cat((x_rgb_p2, x_inf_p2), dim=1))
        x = F.upsample(x, scale_factor=2, mode='nearest') # unpool1
        x = self.decode1(x)
        x = F.upsample(x, scale_factor=2, mode='nearest') # unpool1
        self.resnet_rgb.requires_grad = False
        self.resnet_h.requires_grad = False

        return x


def unit_test():
    import numpy as np
    x = torch.tensor(np.random.rand(2,4,480,640).astype(np.float32))
    model = MFNet(n_class=9)
    y = model(x)
    print('output shape:', y.shape)
    assert y.shape == (2,9,480,640), 'output shape (2,9,480,640) is expected!'
    print('test ok!')


if __name__ == '__main__':
    unit_test()
