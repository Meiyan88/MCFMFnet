import torch
import torch.nn as nn
import torch.nn.functional as F
# from tuSeg.utils.initWeights import init_weights
import numpy as np
from torch.nn import init

from copy import deepcopy
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
import torch
import numpy as np
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
import torch.nn.functional
import SimpleITK as sitk

def weights_init_normal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)



class Sober(nn.Module):
    def __init__(self, channels):
        super(Sober, self).__init__()
        self.channels = channels

        # 定义 Sobel 核心，使用 torch.tensor 直接创建
        self.kernel_x = torch.tensor([[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                                      [[2, 0, -2], [4, 0, -4], [2, 0, -2]],
                                      [[1, 0, -1], [2, 0, -2], [1, 0, -1]]], dtype=torch.float32)

        self.kernel_y = torch.tensor([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                                      [[2, 4, 2], [0, 0, 0], [-2, -4, -2]],
                                      [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]], dtype=torch.float32)

        self.kernel_z = torch.tensor([[[1, 2, 1], [2, 4, 2], [1, 2, 1]],
                                      [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                      [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]], dtype=torch.float32)

    def forward(self, x):
        # 确保卷积核在输入张量相同的设备上
        device = x.device
        sobel_kernel_x = self.kernel_x.unsqueeze(0).expand(self.channels, -1, -1, -1, -1).to(device)
        sobel_kernel_y = self.kernel_y.unsqueeze(0).expand(self.channels, -1, -1, -1, -1).to(device)
        sobel_kernel_z = self.kernel_z.unsqueeze(0).expand(self.channels, -1, -1, -1, -1).to(device)

        # 对每个核应用 3D 卷积
        G_x = F.conv3d(x, sobel_kernel_x, stride=1, padding=1, groups=x.size(1))
        G_y = F.conv3d(x, sobel_kernel_y, stride=1, padding=1, groups=x.size(1))
        # G_z = F.conv3d(x, sobel_kernel_z, stride=1, padding=1, groups=x.size(1))

        # 组合梯度（可以根据需要返回或进一步处理）
        x  = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2) + 1e-6)
        return x


# class Sober(nn.Module):
#     def __init__(self, channels):
#         super(Sober, self).__init__()
#         self.channels = channels
#         self.kernel_x = np.array([[[1, 0, -1], [2, 0, -2], [1, 0, -1]],[[2, 0, -2], [4, 0, -4], [2, 0, -2]],[[1, 0, -1], [2, 0, -2], [1, 0, -1]]], np.float32)#.cuda(1)
#         self.kernel_y = np.array([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],[[2, 4, 2], [0, 0, 0], [-2, -4, -2]],[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]], np.float32)#.cuda(1)
#         self.kernel_z = np.array([[[1, 2, 1], [2, 4, 2], [1, 2, 1]],[[0, 0, 0], [0, 0, 0], [0, 0, 0]],[[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]], np.float32)#.cuda(1)
#
#     def forward(self, x):
#         self.sobel_kernel_x = torch.from_numpy(self.kernel_x).unsqueeze(0).expand(self.channels, 1, 3, 3, 3).float()
#         self.sobel_kernel_y = torch.from_numpy(self.kernel_y).unsqueeze(0).expand(self.channels, 1, 3, 3, 3).float()
#         self.sobel_kernel_z = torch.from_numpy(self.kernel_z).unsqueeze(0).expand(self.channels, 1, 3, 3, 3).float()
#
#         G_x = F.conv3d(x, self.sobel_kernel_x, stride=1, padding=1, groups=x.size(1))
#         G_y = F.conv3d(x, self.sobel_kernel_y, stride=1, padding=1, groups=x.size(1))
#         G_z = F.conv3d(x, self.sobel_kernel_z, stride=1, padding=1, groups=x.size(1))
#         x  = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2) + torch.pow(G_z, 2) + 1e-6)
#         return x

class UnetConv3(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, padding_size=1):
        super(UnetConv3, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, padding=padding_size),
                                       nn.InstanceNorm3d(out_size),
                                       nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, padding=padding_size),
                                       nn.InstanceNorm3d(out_size),
                                       nn.ReLU(inplace=True),)

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs

class SoberConv(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, padding_size=1):
        super(SoberConv, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, padding=padding_size),
                                        nn.InstanceNorm3d(out_size),
                                        nn.LeakyReLU(inplace=True),)
        self.sober = nn.Sequential(nn.Conv3d(out_size,out_size//2,kernel_size=1),
                                        nn.Conv3d(out_size//2,out_size//2,kernel_size=3, padding=1),
                                        nn.InstanceNorm3d(out_size//2),
                                        nn.LeakyReLU(inplace=True),
                                        Sober(out_size//2),
                                        nn.InstanceNorm3d(out_size//2),
                                        nn.LeakyReLU(inplace=True),
                                        nn.Conv3d(out_size//2,out_size,kernel_size=1))
        self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, padding=padding_size),
                                       nn.InstanceNorm3d(out_size),
                                       nn.LeakyReLU(inplace=True),)
        self.sober_out = nn.Conv3d(out_size*2, out_size,kernel_size=1)
        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        sober = self.sober(outputs)
        outputs = torch.cat([outputs, sober], dim=1)
        outputs = self.sober_out(outputs)
        outputs = self.conv2(outputs)
        return outputs

class DilateConv3(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=(3,3,1), padding_size=(1,1,0), init_stride=(1,1,1)):
        super(DilateConv3, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv3d(in_size, in_size//2, kernel_size=1),
                                   nn.Conv3d(in_size//2, in_size//2, kernel_size, init_stride, padding_size),
                                   nn.InstanceNorm3d(in_size//2),
                                   nn.ReLU(inplace=True),
                                   nn.Conv3d(in_size//2, out_size,kernel_size=1))
        self.conv2 = nn.Sequential(nn.Conv3d(in_size,in_size//2,kernel_size=1),
                                   nn.Conv3d(in_size//2, in_size//2, kernel_size, dilation=3, padding=3),
                                   nn.InstanceNorm3d(in_size//2),
                                   nn.ReLU(inplace=True),
                                   nn.Conv3d(in_size//2, out_size,kernel_size=1))
        self.skip = nn.Conv3d(in_size, out_size, 1)

        self.out = nn.Conv3d(out_size*3, out_size, 1)

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(inputs)
        skip = self.skip(inputs)
        return self.out(torch.cat([conv1, conv2, skip], 1))

class UnetUp3_CT(nn.Module):
    def __init__(self, in_size, out_size):
        super(UnetUp3_CT, self).__init__()
        self.conv = SoberConv(in_size + out_size, out_size, kernel_size=3, padding_size=1)
        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('SoberConv') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        return self.conv(torch.cat([inputs1, outputs2], 1))

class UnetUp3_Dilate(nn.Module):
    def __init__(self, in_size, out_size):
        super(UnetUp3_Dilate, self).__init__()
        self.conv = DilateConv3(in_size + out_size, out_size, kernel_size=3, padding_size=1)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('DilateConv3') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        return self.conv(torch.cat([inputs1, outputs2], 1))

class UnetDsv3(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(UnetDsv3, self).__init__()
        self.dsv = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                                 nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=True), )

    def forward(self, input):
        return self.dsv(input)

class SBCNet(SegmentationNetwork):
    DEFAULT_BATCH_SIZE_3D = 2
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320

    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480

    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000  # 505789440


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
        super(SBCNet, self).__init__()
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

        feature_scale = 4
        n_classes = 2
        in_channels = 1

        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv3(in_channels, filters[0], kernel_size=3, padding_size=1)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2)

        self.conv2 = UnetConv3(filters[0], filters[1], kernel_size=3, padding_size=1)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2)

        self.conv3 = UnetConv3(filters[1], filters[2], kernel_size=3, padding_size=1)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2)

        self.conv4 = UnetConv3(filters[2], filters[3], kernel_size=3, padding_size=1)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2)

        self.center = UnetConv3(filters[3], filters[4], kernel_size=3, padding_size=1)

        # upsampling
        self.up1_concat4 = UnetUp3_CT(filters[4], filters[3])
        self.up1_concat3 = UnetUp3_CT(filters[3], filters[2])
        self.up1_concat2 = UnetUp3_CT(filters[2], filters[1])
        self.up1_concat1 = UnetUp3_CT(filters[1], filters[0])

        self.up2_concat4 = UnetUp3_Dilate(filters[4], filters[3])
        self.up2_concat3 = UnetUp3_Dilate(filters[3], filters[2])
        self.up2_concat2 = UnetUp3_Dilate(filters[2], filters[1])
        self.up2_concat1 = UnetUp3_Dilate(filters[1], filters[0])

        self.mid_up_outconv = nn.Conv3d(filters[3], 1, 1)
        self.mid_down_outconv = nn.Conv3d(filters[3], 1, 1)
        self.up_outconv = nn.Conv3d(filters[0], n_classes, 1)
        self.down_outconv = nn.Conv3d(filters[0], n_classes, 1)

        self.final = nn.Conv3d(filters[0]*2, n_classes, 1)

        self.edge4 = UnetDsv3(in_size=filters[3], out_size=filters[0], scale_factor=8)
        self.edge3 = UnetDsv3(in_size=filters[2], out_size=filters[0], scale_factor=4)
        self.edge2 = UnetDsv3(in_size=filters[1], out_size=filters[0], scale_factor=2)
        self.edge1 = nn.Conv3d(in_channels=filters[0], out_channels=filters[0], kernel_size=1)
        self.edge_out = nn.Conv3d(in_channels=filters[0]*4, out_channels=1, kernel_size=1)

        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout21 = nn.Dropout(p=0.3)
        self.dropout22 = nn.Dropout(p=0.3)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.InstanceNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        x = self.maxpool1(conv1)

        conv2 = self.conv2(x)
        x = self.maxpool2(conv2)

        conv3 = self.conv3(x)
        x = self.maxpool3(conv3)

        conv4 = self.conv4(x)
        x = self.maxpool4(conv4)

        x = self.center(x)
        x = self.dropout1(x)

        d1 = self.up1_concat4(conv4, x)
        edge4 = self.edge4(d1)
        mid1 = self.mid_up_outconv(d1)
        d1out = self.up1_concat3(conv3, d1)
        edge3 = self.edge3(d1out)
        d1out = self.up1_concat2(conv2, d1out)
        edge2 = self.edge2(d1out)
        d1out = self.up1_concat1(conv1, d1out)
        edge1 = self.edge1(d1out)
        d1out = self.dropout21(d1out)

        edge_out = self.edge_out(torch.cat([edge4,edge3,edge2,edge1], dim=1))

        d2 = self.up2_concat4(conv4, x)
        mid2 = self.mid_down_outconv(d2)
        d2out = self.up2_concat3(conv3, d2)
        d2out = self.up2_concat2(conv2, d2out)
        d2out = self.up2_concat1(conv1, d2out)
        d2out = self.dropout22(d2out)

        final = self.final(torch.cat([d1out,d2out], dim=1))
        d1out = self.up_outconv(d1out)   # maybe yellow block  sobel
        d2out = self.down_outconv(d2out)  # maybe multi-scale green

        if self._deep_supervision and self.do_ds:
            return final, d1out, d2out, mid1, mid2, edge_out
        else:
            return final

        # return final, d1out, d2out, mid1, mid2, edge_out    # final final-seg         # d1  edgeout-output


if __name__ == "__main__":
    net_num_pool_op_kernel_sizes = [[1, 2, 2],
                              [2, 2, 2],
                              [2, 2, 2],
                              [2, 2, 2],
                              [1, 2, 2]]

    net_conv_kernel_sizes = [[1, 3, 3],   #
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
    network = SBCNet(1, 32, 2,
                 len(net_num_pool_op_kernel_sizes),
                 2, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                 dropout_op_kwargs,
                 net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                 net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True)

    data =torch.randn(2, 1,96, 96, 96)
    # block = DownConv(32, 32, first_stride=[1, 2, 2])
    pred = network(data)
    print(network)
    print(pred[0].shape)
    print(pred[1].shape)
    print(pred[2].shape)
    print(pred[3].shape)
    print(pred[4].shape)

    # print(pred.shape)