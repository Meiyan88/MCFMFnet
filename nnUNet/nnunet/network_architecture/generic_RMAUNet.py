import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
import torch
import numpy as np
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
import torch.nn.functional
import SimpleITK as sitk


from torchvision.utils import make_grid, save_image

class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv3DBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class ResSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResSEBlock, self).__init__()
        self.conv1 = Conv3DBlock(in_channels, out_channels)
        self.conv2 = Conv3DBlock(out_channels, out_channels)
        self.se = SEBlock(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        out += self.shortcut(x)
        return F.relu(out)


class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DilatedConvBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=4, dilation=4)
        self.conv4 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=8, dilation=8)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1 = self.relu(self.conv1(x))
        c2 = self.relu(self.conv2(x))
        c3 = self.relu(self.conv3(x))
        c4 = self.relu(self.conv4(x))
        return torch.cat([c1, c2, c3, c4], dim=1)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc1(self.avg_pool(x).view(x.size(0), -1))
        avg_out = self.fc2(self.relu(avg_out))

        max_out = self.fc1(self.max_pool(x).view(x.size(0), -1))
        max_out = self.fc2(self.relu(max_out))

        out = avg_out + max_out
        return x * self.sigmoid(out).view(x.size(0), x.size(1), 1, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv3d = nn.Conv3d(2, 1024, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv3d(x)
        return x * self.sigmoid(x)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        self.dilated_convs = DilatedConvBlock(in_channels, out_channels // 4)
        self.channel_attention = ChannelAttention(out_channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.dilated_convs(x)
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class RMAUNet(SegmentationNetwork):
    def __init__(self, inchannel=1, outchannel=2):
        super(RMAUNet, self).__init__()
        self.nonlin = nn.LeakyReLU
        self.nonlin_kwargs = None
        self.dropout_op_kwargs = None
        self.norm_op_kwargs = None
        self.weightInitializer = InitWeights_He(1e-2)
        self.conv_op = nn.Conv3d
        self.norm_op = nn.InstanceNorm3d
        self.dropout_op = nn.Dropout3d
        self.num_classes = 2
        self.final_nonlin = lambda x: x
        self._deep_supervision = False
        self.do_ds = False


        # Encoder path
        self.down1 = ResSEBlock(1, 64)
        self.pool1 = nn.MaxPool3d(2)
        self.down2 = ResSEBlock(64, 128)
        self.pool2 = nn.MaxPool3d(2)
        self.down3 = ResSEBlock(128, 256)
        self.pool3 = nn.MaxPool3d(2)
        self.down4 = ResSEBlock(256, 512)
        self.pool4 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = Bottleneck(512, 1024)

        # Decoder path
        self.up3 = nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2)
        self.dec3 = ResSEBlock(1024, 512)
        self.up2 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.dec2 = ResSEBlock(512, 256)
        self.up1 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec1 = ResSEBlock(256, 128)
        self.up0 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec0 = ResSEBlock(128, 64)

        # Final Convolution
        self.final_conv = nn.Conv3d(64, 2, kernel_size=1)

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        d4 = self.down4(p3)
        p4 = self.pool4(d4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder
        up3 = self.up3(b)
        cat3 = torch.cat((up3, d4), dim=1)
        dec3 = self.dec3(cat3)

        up2 = self.up2(dec3)
        cat2 = torch.cat((up2, d3), dim=1)
        dec2 = self.dec2(cat2)

        up1 = self.up1(dec2)
        cat1 = torch.cat((up1, d2), dim=1)
        dec1 = self.dec1(cat1)

        up0 = self.up0(dec1)
        cat0 = torch.cat((up0, d1), dim=1)
        dec0 = self.dec0(cat0)

        # Final convolution
        out = self.final_conv(dec0)
        return out


if __name__ == "__main__":
    conv_op = nn.Conv3d
    dropout_op = nn.Dropout3d
    norm_op = nn.InstanceNorm3d  # elif len(self.patch_size) == 3:  self.threeD = True

    norm_op_kwargs = {'eps': 1e-5, 'affine': True}
    dropout_op_kwargs = {'p': 0, 'inplace': True}
    net_nonlin = nn.LeakyReLU
    net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
    # Initialize the model

    network = RMAUNet(1,2).cuda(1)

    data =torch.randn(1, 1, 128, 128, 128).cuda(1)
    # block = DownConv(32, 32, first_stride=[1, 2, 2])
    print(network)
    pred = network(data)
    print(pred.shape)


