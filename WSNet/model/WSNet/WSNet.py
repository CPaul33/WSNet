import torch
import torch.nn as nn
import torch.nn.functional as F

from .CSHA import *


class _FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),
            nn.Dropout(0.05),
            nn.Conv2d(inter_channels, out_channels, 1, 1, 0)
        )

    def forward(self, x):
        return self.block(x)


def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU(),
        nn.Dropout(0.1)
    )


class WEM(nn.Module):
    def __init__(self, in_channels):
        super(WEM, self).__init__()
        inter_channels = int(in_channels / 4)
        out_channels = in_channels
        self.conv_1 = nn.Sequential( #RF1*1
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
        self.conv_2 = nn.Sequential( #RF3*3
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, bias=True),
            # nn.BatchNorm2d(inter_channels)
            nn.LeakyReLU(),
            nn.Conv2d(inter_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
        self.conv_3 = nn.Sequential( #RF5*5
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, bias=True),
            # nn.BatchNorm2d(inter_channels)
            nn.LeakyReLU(),
            nn.Conv2d(inter_channels, out_channels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
        self.conv_4 = nn.Sequential( #RF7*7
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, bias=True),
            # nn.BatchNorm2d(inter_channels)
            nn.LeakyReLU(),
            nn.Conv2d(inter_channels, out_channels, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
        self.conv_5 = nn.Sequential(  # RF13*13
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, bias=True),
            # nn.BatchNorm2d(inter_channels)
            nn.LeakyReLU(),
            nn.Conv2d(inter_channels, out_channels, kernel_size=5, stride=1, padding=6, dilation=3),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
        self.conv_6 = nn.Sequential(  # RF17*17
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, bias=True),
            # nn.BatchNorm2d(inter_channels)
            nn.LeakyReLU(),
            nn.Conv2d(inter_channels, out_channels, kernel_size=5, stride=1, padding=8, dilation=4),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        out1 = self.conv_1(x)
        out2 = self.conv_2(x)
        out3 = self.conv_3(x)
        out4 = self.conv_4(x)
        out5 = self.conv_5(x)
        out6 = self.conv_6(x)

        out = out1 + out2 + out3 + out4 + out5 + out6
        # out = out1 + out2 + out3 + out4
        return out


class WSNet(nn.Module):
    def __init__(self):
        super(WSNet, self).__init__()
        self.conv1 = conv_batch(1, 16)
        self.conv2 = conv_batch(16, 32, stride=2)
        self.wem = WEM(in_channels=32)
        self.csha = CSHA(32, 32)
        self.conv_ = conv_batch(32, 32, 3, padding=1)
        self.conv_res = conv_batch(16, 32, 1, padding=0)
        self.leakyrelu = nn.LeakyReLU(True)
        self.head = _FCNHead(32, 1)

    def forward(self, x):
        _, _, h, w = x.shape

        out1 = self.conv1(x) #x'
        out2 = self.conv2(out1) #x_l

        out2 = self.wem(out2) #x_w

        out2 = self.conv_(out2) #x_w
        out2 = self.csha(out2) #x_g

        temp = F.interpolate(out2, size=[h, w], mode='bilinear') #x_h
        temp2 = self.conv_res(out1) #x_r

        # x_new = temp + temp2
        out = self.leakyrelu(temp + temp2) #x_f
        pred = self.head(out) #x_fcn

        return pred.sigmoid()


class WSNet_NUDT(nn.Module):
    def __init__(self):
        super(WSNet_NUDT, self).__init__()
        self.conv = conv_batch(1, 32)
        self.conv1 = conv_batch(32, 64)
        self.conv2 = conv_batch(64, 128, stride=2)
        self.conv3 = conv_batch(128, 64)
        self.wem = WEM(in_channels=128)
        self.csha = CSHA(128, 32)
        self.conv_ = conv_batch(128, 128, 3, padding=1)
        self.conv_res1 = conv_batch(32, 128, 1, padding=0)
        self.conv_res2 = conv_batch(64, 128, 1, padding=0)
        self.relu = nn.ReLU()
        self.head = _FCNHead(64, 1)

    def forward(self, x):
        _, _, h, w = x.shape
        x = self.conv(x)
        out1 = self.conv1(x)  # x'
        out2 = self.conv2(out1)  # x_l

        out2 = self.wem(out2)  # x_w

        out2 = self.conv_(out2)  # x_w
        out2 = self.csha(out2)  # x_g

        temp = F.interpolate(out2, size=[h, w], mode='bilinear')  # x_h
        temp2 = self.conv_res2(out1)  # x_r

        # x_new = temp + temp2
        out = self.relu(temp + temp2)  # x_f
        x = self.conv_res1(x)
        out = self.conv3(self.relu(out + x))
        pred = self.head(out)  # x_fcn

        return pred.sigmoid()
