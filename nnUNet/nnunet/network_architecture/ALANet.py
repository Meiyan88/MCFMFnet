import torch
import torch.nn as nn
import numpy as np
import os

from copy import deepcopy
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
import torch
import numpy as np
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
import torch.nn.functional
import SimpleITK as sitk
import torch.nn.functional as F

from torchvision.utils import make_grid, save_image



class WRRB(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(WRRB, self).__init__()
        self.conv111 = nn.Conv3d(inchannel, outchannel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv331 = nn.Conv3d(outchannel, outchannel, kernel_size=[3,3,1], stride=1, padding=[1,1,0], bias=False)
        self.conv113 = nn.Conv3d(outchannel, outchannel, kernel_size=[1,1,3], stride=1, padding=[0,0,1], bias=False)

        self.sec_conv111 = nn.Conv3d(outchannel, outchannel, kernel_size=1, stride=1, padding=0, bias=False)
        self.thr_conv111 = nn.Conv3d(outchannel, outchannel, kernel_size=1, stride=1, padding=0, bias=False)

        self.res_conv111 = nn.Conv3d(inchannel, outchannel, kernel_size=1, stride=1, padding=0, bias=False)

        self.reluconv = nn.Conv3d(3*outchannel, outchannel, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_branch1 = x
        x_branch2 = self.conv111(x)
        x_branch2 = torch.cat([self.sec_conv111(self.conv331(x_branch2)), self.thr_conv111(self.conv113(x_branch2))], dim=1)
        x_branch3 = self.res_conv111(x)
        x_cat = torch.cat([x_branch2, x_branch3], dim=1)
        x_out = x_branch1 + self.relu(self.reluconv(x_cat))

        return x_out

class RM(nn.Module):
    def __init__(self, inchannel):
        super(RM, self).__init__()
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.conv = nn.Conv3d(inchannel, inchannel, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x_pool = self.maxpool(x)
        x_conv = self.conv(x)
        x_cat = torch.cat([x_pool, x_conv], dim=1)
        return x_cat

class WSB(nn.Module):
    def __init__(self, num_input_features, growth_rate=16, num_convolutions=4):
        super(WSB, self).__init__()
        self.layers = nn.ModuleList()

        for _ in range(num_convolutions):
            self.layers.append(ConvBlock3D(num_input_features, growth_rate))
            num_input_features += growth_rate  # Update input features for the next layer
        # 添加一个1x1卷积层，确保输入和输出通道一致
        self.final_conv = nn.Conv3d(num_input_features, num_input_features - (num_convolutions * growth_rate), kernel_size=1)

    def forward(self, x):
        original_input = x  # 保存原始输入
        for layer in self.layers:
            out = layer(x)
            x = torch.cat([x, out], dim=1)  # Concatenate along the channel dimension

        x = self.final_conv(x)  # 通过1x1卷积调整通道数
        x += original_input  # 添加原始输入以实现跳跃连接
        return x


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(ConvBlock3D, self).__init__()
        self.instancenorm = nn.InstanceNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = self.instancenorm(x)
        x = self.relu(x)
        x = self.conv(x)
        return x

class H3DC1(nn.Module):
    def __init__(self, inchannel):
        super(H3DC1, self).__init__()

        # 定义三个3D空洞卷积层，膨胀率分别为1, 2, 3，步长为1
        self.conv1 = nn.Conv3d(inchannel, inchannel, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2 = nn.Conv3d(inchannel, inchannel, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv3 = nn.Conv3d(inchannel, inchannel, kernel_size=3, stride=1, padding=3, dilation=3)


        self.conv = nn.Conv3d(3*inchannel, inchannel, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn = nn.BatchNorm3d(inchannel)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        # 通过三个卷积层并获取输出
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)


        out = self.relu(self.bn(self.conv(torch.cat([out1, out2, out3], dim=1))))
        # 将输出沿通道维度拼接
        return out  # dim=1表示沿着通道维度拼接


class H3DC2(nn.Module):
    def __init__(self, inchannel):
        super(H3DC2, self).__init__()

        # 定义三个3D空洞卷积层，膨胀率分别为1, 2, 3，步长为1
        self.conv1 = nn.Conv3d(inchannel, inchannel, kernel_size=3, stride=1, padding=3, dilation=3)
        self.conv2 = nn.Conv3d(inchannel, inchannel, kernel_size=3, stride=1, padding=4, dilation=4)
        self.conv3 = nn.Conv3d(inchannel, inchannel, kernel_size=3, stride=1, padding=5, dilation=5)

        self.conv = nn.Conv3d(3 * inchannel, inchannel, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn = nn.BatchNorm3d(inchannel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 通过三个卷积层并获取输出
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)

        out = self.relu(self.bn(self.conv(torch.cat([out1, out2, out3], dim=1))))
        # 将输出沿通道维度拼接
        return out  # dim=1表示沿着通道维度拼接

class TL(nn.Module):
    def __init__(self, inchannel):
        super(TL, self).__init__()
        self.conv = nn.Conv3d(inchannel, 2*inchannel, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm3d(2*inchannel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out

class initconv(nn.Module):
    def __init__(self, inchannel):
        super(initconv, self).__init__()
        self.conv = nn.Conv3d(inchannel, 32, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv(x)
        return out

class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)


class ChannelAttention3D(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc1 = nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return torch.sigmoid(avg_out + max_out)


class SpatialAttention3D(nn.Module):
    def __init__(self):
        super(SpatialAttention3D, self).__init__()
        self.conv1 = nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat((avg_out, max_out), dim=1)
        return torch.sigmoid(self.conv1(concat))


class LAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(LAM, self).__init__()
        self.convdown = nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.convbn = nn.BatchNorm3d(32)
        self.convrelu = nn.ReLU()

        self.channel_attention = ChannelAttention3D(32, reduction)
        self.spatial_attention = SpatialAttention3D()

        self.conv = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.convbn2 = nn.BatchNorm3d(32)
        self.convrelu2 = nn.ReLU()

    def forward(self, x):
        x = self.convrelu(self.convbn(self.convdown(x)))
        # 计算通道注意力
        channel_weights = self.channel_attention(x)
        # 将输入特征图与通道注意力权重相乘
        x_c = x * channel_weights + x

        # 计算空间注意力
        spatial_weights = self.spatial_attention(x)
        # 将输入特征图与空间注意力权重相乘
        x_s = x * spatial_weights + x

        x_out = self.convrelu2(self.convbn2(self.conv(x_c + x_s)))
        return x_out


class PAM(nn.Module):
    def __init__(self, in_channels):
        super(PAM, self).__init__()
        self.conv0 = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3 = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.maxpool = nn.AdaptiveMaxPool3d(output_size=(1, 1, 1))


        self.maxconv1 = nn.Conv3d(in_channels, in_channels // 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.avgconv1 = nn.Conv3d(in_channels, in_channels // 4, kernel_size=1, stride=1, padding=0, bias=False)

        self.maxrelu = nn.ReLU()
        self.avgrelu = nn.ReLU()

        self.maxconv2 = nn.Conv3d(in_channels // 4, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.avgconv2 = nn.Conv3d(in_channels // 4, in_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.convoutput = nn.Conv3d(in_channels, 2, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, AF0, AF1, AF2, AF3):

        AFl = self.conv0(AF0) + self.conv1(AF1) + self.conv2(AF2) + self.conv3(AF3)
        AFlmax = self.maxpool(AFl)
        AFlavg = self.avgpool(AFl)

        AFadd = self.maxconv2(self.maxrelu(self.maxconv1(AFlmax))) + self.avgconv2(self.avgrelu(self.avgconv1(AFlavg)))
        Ml = self.sigmoid(AFadd)

        outputs = self.convoutput(AFl * Ml)

        return outputs


class ALA_Net(SegmentationNetwork):
    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=None,
                 seg_output_use_bias=False):
        super(ALA_Net, self).__init__()
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
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision



        self.WRRB1 = WRRB(input_channels, 32)
        self.RM1 = RM(32)

        self.WRRB2 = WRRB(64, 64)
        self.RM2 = RM(64)

        self.WRRB3 = WRRB(128, 128)
        self.RM3 = RM(128)

        self.WRRB4 = WRRB(256, 256)

        self.initconv = initconv(input_channels)
        self.WSB1 = WSB(32)
        self.TL1 = TL(32)

        self.WSB2 = WSB(64)
        self.TL2 = TL(64)

        self.H3DC1 = H3DC1(128)

        self.H3DC2 = H3DC2(128)

        self.upx2 = Upsample(scale_factor=[2, 2, 2], mode='trilinear')
        self.upx4 = Upsample(scale_factor=[4, 4, 4], mode='trilinear')

        self.F0conv = nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1)
        self.F0bn = nn.BatchNorm3d(32)
        self.F0relu = nn.ReLU()

        self.F1conv = nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1)
        self.F1bn = nn.BatchNorm3d(64)
        self.F1relu = nn.ReLU()

        self.F2conv = nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=1)
        self.F2bn = nn.BatchNorm3d(128)
        self.F2relu = nn.ReLU()

        self.F3conv = nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=1)
        self.F3bn = nn.BatchNorm3d(128)
        self.F3relu = nn.ReLU()

        self.transpose = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2, bias=False)

        self.Fcusionconv = nn.Conv3d(160, 32, kernel_size=3, stride=1, padding=1)
        self.Fcusionbn = nn.BatchNorm3d(32)
        self.Fcusionrelu = nn.ReLU()


        self.convout0 = nn.Conv3d(32, 2, kernel_size=1, stride=1, padding=0)
        self.convout1 = nn.Conv3d(32, 2, kernel_size=1, stride=1, padding=0)
        self.convout2 = nn.Conv3d(32, 2, kernel_size=1, stride=1, padding=0)
        self.convout3 = nn.Conv3d(32, 2, kernel_size=1, stride=1, padding=0)

        self.LAM0 = LAM(64)
        self.LAM1 = LAM(96)
        self.LAM2 = LAM(160)
        self.LAM3 = LAM(160)

        self.PAM = PAM(32)

    def forward(self, x):
        GCE_x1 = self.WRRB1(x)

        GCE_x2 = self.RM1(GCE_x1)
        GCE_x2 = self.WRRB2(GCE_x2)

        GCE_x3 = self.RM2(GCE_x2)
        GCE_x3 = self.WRRB3(GCE_x3)

        GCE_x4 = self.RM3(GCE_x3)
        GCE_x4 = self.WRRB4(GCE_x4)



        DSRP_x1 = self.initconv(x)
        DSRP_x1 = self.WSB1(DSRP_x1)

        DSRP_x2 = self.TL1(DSRP_x1)
        DSRP_x2 = self.WSB2(DSRP_x2)


        DSRP_x3 = self.TL2(DSRP_x2)
        DSRP_x3 = self.H3DC1(DSRP_x3)

        DSRP_x4 = self.H3DC2(DSRP_x3)


        # middle fusion part
        F0 = self.F0relu(self.F0bn(self.F0conv(torch.cat([GCE_x1, DSRP_x1], dim=1))))
        F1 = self.F1relu(self.F1bn(self.F1conv(torch.cat([self.upx2(GCE_x2), self.upx2(DSRP_x2)], dim=1))))
        F2 = self.F2relu(self.F2bn(self.F2conv(torch.cat([self.upx4(GCE_x3), self.upx4(DSRP_x3)], dim=1))))
        F3 = self.F3relu(self.F3bn(self.F3conv(torch.cat([self.upx4(self.transpose(GCE_x4)), self.upx4(DSRP_x4)], dim=1))))

        FML = self.Fcusionconv(torch.cat([F0, F3], dim=1))

        del DSRP_x1, DSRP_x2, DSRP_x3, DSRP_x4
        del GCE_x1, GCE_x2, GCE_x3, GCE_x4

        M0 = torch.cat([F0, FML], dim=1)
        M1 = torch.cat([F1, FML], dim=1)
        M2 = torch.cat([F2, FML], dim=1)
        M3 = torch.cat([F3, FML], dim=1)

        AF0 = self.LAM0(M0)
        AF1 = self.LAM1(M1)
        AF2 = self.LAM2(M2)
        AF3 = self.LAM3(M3)

        del M0, M1, M2, M3


        segout0 = self.convout0(AF0)
        segout1 = self.convout1(AF1)
        segout2 = self.convout2(AF2)
        segout3 = self.convout3(AF3)

        out_r = self.PAM(AF0, AF1, AF2, AF3)

        del AF0, AF1, AF2, AF3

        if self._deep_supervision and self.do_ds:
            return [out_r, segout0, segout1, segout2, segout3]
        else:
            return out_r



if __name__ == "__main__":

    net_num_pool_op_kernel_sizes = [[1, 2, 2],
                                    [2, 2, 2],
                                    [2, 2, 2],
                                    [2, 2, 2],
                                    [1, 2, 2]]

    net_conv_kernel_sizes = [[1, 3, 3],  #
                             [3, 3, 3],
                             [3, 3, 3],
                             [3, 3, 3],
                             [3, 3, 3],
                             [3, 3, 3]]
    conv_op = nn.Conv3d
    dropout_op = nn.Dropout3d
    norm_op = nn.InstanceNorm3d  # elif len(self.patch_size) == 3:  self.threeD = True

    norm_op_kwargs = {'eps': 1e-5, 'affine': True}
    dropout_op_kwargs = {'p': 0, 'inplace': True}
    net_nonlin = nn.LeakyReLU
    net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
    network = ALA_Net(1, 32, 2,
                           len(net_num_pool_op_kernel_sizes),
                           2, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                           dropout_op_kwargs,
                           net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                           net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True)

    data =torch.randn(2, 1, 48, 192, 192)
    # block = DownConv(32, 32, first_stride=[1, 2, 2])
    print(network)
    pred = network(data)
    # print(pred.shape)
    print(pred[0].shape)
    print(pred[1].shape)
    print(pred[2].shape)
    print(pred[3].shape)
    print(pred[4].shape)



