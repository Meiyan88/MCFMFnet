#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


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

class ConvDropoutNormNonlin(nn.Module):
    """
    fixes a bug in ConvDropoutNormNonlin where lrelu was used regardless of nonlin. Bad.
    """

    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs  # ReLU params
        self.nonlin = nonlin  # leakyrelu

        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs # dropout ratio

        self.norm_op_kwargs = norm_op_kwargs
        self.norm_op = norm_op

        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op


        # look here   pass kind of conv_op  and init it with **self.conv_kwargs
        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)  # conv with its kwargs
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)    #id dropout and ratio not None then give p=0.2 maybe
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)   # it says use InstanceNorm
        self.lrelu = self.nonlin(**self.nonlin_kwargs)     #nonlin here it's leakyrelu

    def forward(self, x):
        x = self.conv(x)   # do conv operation
        if self.dropout is not None:  # if need dropout then do dropout
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))  #then do InstanceNorm and relu operation!
        # conv -> (Dropout) -> BatchNorm -> ReLU

class ConvDropoutNonlinNorm(ConvDropoutNormNonlin):
    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.instnorm(self.lrelu(x))    #  just exchange the position of BN and ReLU


class StackedConvLayers(nn.Module):   # This we maybe DoubleConv just Stack block
    def __init__(self, input_feature_channels, output_feature_channels, num_convs,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None, basic_block=ConvDropoutNormNonlin):  # here is the basic_block
        '''
        stacks ConvDropoutNormLReLU layers. initial_stride will only be applied to first layer in the stack. The other parameters affect all layers
        :param input_feature_channels:
        :param output_feature_channels:
        :param num_convs:
        :param dilation:
        :param kernel_size:
        :param padding:
        :param dropout:
        :param initial_stride:
        :param conv_op:
        :param norm_op:
        :param dropout_op:
        :param inplace:
        :param neg_slope:
        :param norm_affine:
        :param conv_bias:
        '''
        self.input_channels = input_feature_channels     # here in and out channels
        self.output_channels = output_feature_channels

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin

        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs

        self.norm_op_kwargs = norm_op_kwargs
        self.norm_op = norm_op

        self.conv_kwargs = conv_kwargs   # default stride=1
        self.conv_op = conv_op


        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)  # default stride=1
            self.conv_kwargs_first_conv['stride'] = first_stride
        else:
            self.conv_kwargs_first_conv = conv_kwargs  # i_1  i_2  just for i_1 diy kwargs

        super(StackedConvLayers, self).__init__()
        # a bit different in the first each block conv:Stride!!!
        self.blocks = nn.Sequential(
            *([basic_block(input_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs_first_conv,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs)] +
              [basic_block(output_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs) for _ in range(num_convs - 1)]))  # Attn  look at self.conv_kwargs

    def forward(self, x):  # total nums: num_convs
        return self.blocks(x)  # dispereate the first conv block then stack num_convs -1 conv blocks


def print_module_training_status(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d) or isinstance(module, nn.Dropout3d) or \
            isinstance(module, nn.Dropout2d) or isinstance(module, nn.Dropout) or isinstance(module, nn.InstanceNorm3d) \
            or isinstance(module, nn.InstanceNorm2d) or isinstance(module, nn.InstanceNorm1d) \
            or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d) or isinstance(module,
                                                                                                      nn.BatchNorm1d):
        print(str(module), module.training)


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


class RAG_attention(nn.Module):
    def __init__(self, in_channels, gating_channels, conv_kernels, pool_kernels):
        super(RAG_attention, self).__init__()
        self.softmax = softmax_helper
        # self.conv = nn.Conv3d(in_channels, 2, 1, 1, 0, 1, 1, False)
        # self.threshold = 0.975  # ori 0.95
        self.in_channels = in_channels
        self.gating_channels = gating_channels

        self.sigmoid = nn.Sigmoid()
        conv_op = nn.Conv3d
        norm_op = nn.InstanceNorm3d
        dropout_op = nn.Dropout3d
        nonlin = nn.LeakyReLU

        nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True} # true inplace
        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}

        # conv_kwargs = {'kernel_size': 1, 'stride': 1, 'padding': 0, 'dilation': 1, 'bias': True}
        # self.conv1 = ConvDropoutNormNonlin(in_channels, in_channels, conv_op, conv_kwargs, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs)

        conv_kwargs = {'kernel_size': conv_kernels, 'stride': pool_kernels, 'padding': 0, 'dilation': 1, 'bias': False}
        self.theta = ConvDropoutNormNonlin(in_channels, in_channels, conv_op, conv_kwargs, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs)

        conv_kwargs = {'kernel_size': 1, 'stride': 1, 'padding': 0, 'dilation': 1, 'bias': True}
        self.phi = ConvDropoutNormNonlin(gating_channels, in_channels, conv_op, conv_kwargs, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs)

        conv_kwargs = {'kernel_size': 1, 'stride': 1, 'padding': 0, 'dilation': 1, 'bias': True}
        self.psi = ConvDropoutNormNonlin(in_channels, 1, conv_op, conv_kwargs, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs)

        conv_kwargs = {'kernel_size': 1, 'stride': 1, 'padding': 0, 'dilation': 1, 'bias': True}
        self.W = ConvDropoutNormNonlin(in_channels, in_channels, conv_op, conv_kwargs, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs)
                    # fusion channel conv1

        self.lrelu = nonlin(**nonlin_kwargs)
        self.upsample_mode = 'trilinear'

    def forward(self, x, g):
        # x:skip feature
        # g:upsample feature
        # g = self.conv(y)  # 320

        input_size = x.size()  # skip-conn
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)  # we need x-down sample
        theta_x_size = theta_x.size() # down-x

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = nn.functional.interpolate(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)

        f = theta_x + phi_g
        f = self.lrelu(f)

        sigm_psi_f = self.sigmoid(self.psi(f))
        RA_sigm_psi_f = 1 - sigm_psi_f  # Reverse attention map

        # upsample the attentions and multiply
        RA_sigm_psi_f = nn.functional.interpolate(RA_sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = RA_sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y

class TEM(nn.Module):
    def __init__(self, inchannel, outchannel=32):
        super(TEM, self).__init__()
        conv_op = nn.Conv3d
        norm_op = nn.InstanceNorm3d
        dropout_op = nn.Dropout3d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        conv_kwargs = {'kernel_size': 1, 'stride': 1, 'padding': 0, 'dilation': 1, 'bias': True}
        self.conv_res = ConvDropoutNormNonlin(inchannel, outchannel, conv_op, conv_kwargs, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs)

        conv_kwargs = {'kernel_size': 1, 'stride': 1, 'padding': 0, 'dilation': 1, 'bias': True}
        self.conv1x1 = ConvDropoutNormNonlin(inchannel, outchannel, conv_op, conv_kwargs, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs)

        conv_kwargs = {'kernel_size': 1, 'stride': 1, 'padding': 0, 'dilation': 1, 'bias': True}
        self.conv_1x3x3 = ConvDropoutNormNonlin(inchannel, outchannel, conv_op, conv_kwargs, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs)
        conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}
        self.conv3x3 = ConvDropoutNormNonlin(outchannel, outchannel, conv_op, conv_kwargs, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs)

        conv_kwargs = {'kernel_size': 1, 'stride': 1, 'padding': 0, 'dilation': 1, 'bias': True}
        self.conv_1x5x5 = ConvDropoutNormNonlin(inchannel, outchannel, conv_op, conv_kwargs, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs)
        conv_kwargs = {'kernel_size': 5, 'stride': 1, 'padding': 2, 'dilation': 1, 'bias': True}
        self.conv5x5 = ConvDropoutNormNonlin(outchannel, outchannel, conv_op, conv_kwargs, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs)

        conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}
        self.conv3x3_post = ConvDropoutNormNonlin(outchannel*3, outchannel, conv_op, conv_kwargs, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs)

        self.relu = nn.LeakyReLU(True)

    def forward(self, x):
        # here we may need sobel convolution feature
        x1 = self.conv1x1(x)  # 128 --> 32

        x2 = self.conv_1x3x3(x) # 256 --> 32
        x2 = self.conv3x3(x2)  # 32 --> 32

        x3 = self.conv_1x5x5(x)  # 320 --> 32
        x3 = self.conv5x5(x3)  # 32 --> 32

        x_cat = self.conv3x3_post(torch.cat((x1, x2, x3), dim=1)) # 96 --> 32

        x = self.relu(x_cat + self.conv_res(x))  # 32
        return x

class PDC(nn.Module):
    def __init__(self, inchannel=32):
        super(PDC, self).__init__()
        conv_op = nn.Conv3d
        norm_op = nn.InstanceNorm3d
        dropout_op = nn.Dropout3d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        conv_kwargs = {'kernel_size': 1, 'stride': 1, 'padding': 0, 'dilation': 1, 'bias': True}
        self.upsample = Upsample(scale_factor=2, mode='trilinear')
        self.conv_upsample1 = ConvDropoutNormNonlin(inchannel, inchannel, conv_op, conv_kwargs, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs)
        self.conv_upsample2 = ConvDropoutNormNonlin(inchannel, inchannel, conv_op, conv_kwargs, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs)
        self.conv_upsample3 = ConvDropoutNormNonlin(inchannel, inchannel, conv_op, conv_kwargs, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs)
        self.conv_upsample4 = ConvDropoutNormNonlin(inchannel, inchannel, conv_op, conv_kwargs, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs)
        self.conv_upsample5 = ConvDropoutNormNonlin(2*inchannel, 2*inchannel, conv_op, conv_kwargs, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs)

        self.conv_concat2 = ConvDropoutNormNonlin(2*inchannel, 2*inchannel, conv_op, conv_kwargs, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs)
        self.conv_concat3 = ConvDropoutNormNonlin(3*inchannel, 3*inchannel, conv_op, conv_kwargs, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs)

        self.conv4 = ConvDropoutNormNonlin(3*inchannel, 3*inchannel, conv_op, conv_kwargs, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs)

    def forward(self, x1, x2, x3):
        # x5 --> x4 --> x3
        x1_1 = x1 # x5 smallest
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2 # x4 middle
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) * self.conv_upsample3(self.upsample(x2)) * x3
        # different with NCD is how to make x3_1
        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)  # 96 ---> 96
        # x = self.conv5(x)

        return x

class meta_fusion(nn.Module):
    def __init__(self, in_channel=96):
        super(meta_fusion, self).__init__()
        conv_op = nn.Conv3d
        norm_op = nn.InstanceNorm3d
        dropout_op = nn.Dropout3d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}
        self.conv3x3 = ConvDropoutNormNonlin(416, 320, conv_op, conv_kwargs, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs)

    def forward(self, x, y):
        x = self.conv3x3(torch.cat((x, y),dim=1))

        return x


class Generic_UNet(SegmentationNetwork):
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
                 max_num_features=None, basic_block=ConvDropoutNormNonlin,
                 seg_output_use_bias=False):
        """
        basically more flexible than v1, architecture is the same

        Does this look complicated? Nah bro. Functionality > usability

        This does everything you need, including world peace.

        Questions? -> f.isensee@dkfz.de
        """
        super(Generic_UNet, self).__init__()
        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.upscale_logits = upscale_logits
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision

        if conv_op == nn.Conv2d:
            upsample_mode = 'bilinear'
            pool_op = nn.MaxPool2d
            transpconv = nn.ConvTranspose2d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3)] * (num_pool + 1)
        elif conv_op == nn.Conv3d:
            upsample_mode = 'trilinear'
            pool_op = nn.MaxPool3d
            transpconv = nn.ConvTranspose3d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        if max_num_features is None:
            if self.conv_op == nn.Conv3d:
                self.max_num_features = self.MAX_NUM_FILTERS_3D
            else:
                self.max_num_features = self.MAX_FILTERS_2D
        else:
            self.max_num_features = max_num_features

        self.conv_blocks_context = []
        self.conv_blocks_localization = []
        self.td = []
        self.tu = []
        # self.raa_tu = []

        self.RAG = []
        self.TEM_list = []
        self.seg_outputs = []

        output_features = base_num_features
        input_features = input_channels

        for d in range(num_pool):
            # determine the first stride
            if d != 0 and self.convolutional_pooling:
                first_stride = pool_op_kernel_sizes[d - 1]
            else:
                first_stride = None

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            # add convolutions
            self.conv_blocks_context.append(StackedConvLayers(input_features, output_features, num_conv_per_stage,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride, basic_block=basic_block))
            self.TEM_list.append(TEM(output_features))

            if not self.convolutional_pooling:
                self.td.append(pool_op(pool_op_kernel_sizes[d]))
            input_features = output_features
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))

            output_features = min(output_features, self.max_num_features)

        # now the bottleneck.
        # determine the first stride
        if self.convolutional_pooling:
            first_stride = pool_op_kernel_sizes[-1]
        else:
            first_stride = None

        # the output of the last conv must match the number of features from the skip connection if we are not using
        # convolutional upsampling. If we use convolutional upsampling then the reduction in feature maps will be
        # done by the transposed conv
        if self.convolutional_upsampling:
            final_num_features = output_features
        else:
            final_num_features = self.conv_blocks_context[-1].output_channels

        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]
        self.conv_blocks_context.append(nn.Sequential(
            StackedConvLayers(input_features, output_features, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, first_stride, basic_block=basic_block),
            StackedConvLayers(output_features, final_num_features, 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, basic_block=basic_block)))

        # if we don't want to do dropout in the localization pathway then we set the dropout prob to zero here
        if not dropout_in_localization:
            old_dropout_p = self.dropout_op_kwargs['p']
            self.dropout_op_kwargs['p'] = 0.0

        # now lets build the localization pathway
        for u in range(num_pool):
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context[
                -(2 + u)].output_channels  # self.conv_blocks_context[-1] is bottleneck, so start with -2
            n_features_after_tu_and_concat = nfeatures_from_skip * 2

            # the first conv reduces the number of features to match those of skip
            # the following convs work on that number of features
            # if not convolutional upsampling then the final conv reduces the num of features again
            if u != num_pool - 1 and not self.convolutional_upsampling:
                final_num_features = self.conv_blocks_context[-(3 + u)].output_channels
            else:
                final_num_features = nfeatures_from_skip

            if not self.convolutional_upsampling:
                self.tu.append(Upsample(scale_factor=pool_op_kernel_sizes[-(u + 1)], mode=upsample_mode))
                # self.raa_tu.append(Upsample(scale_factor=pool_op_kernel_sizes[-(u + 1)], mode=upsample_mode))
            else:
                self.tu.append(transpconv(nfeatures_from_down, nfeatures_from_skip, pool_op_kernel_sizes[-(u + 1)],
                                          pool_op_kernel_sizes[-(u + 1)], bias=False))
                # self.raa_tu.append(Upsample(scale_factor=pool_op_kernel_sizes[-(u + 1)], mode=upsample_mode))

            self.RAG.append(RAG_attention(nfeatures_from_skip, nfeatures_from_down, pool_op_kernel_sizes[-(u + 1)], pool_op_kernel_sizes[-(u + 1)]))

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u + 1)]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u + 1)]
            self.conv_blocks_localization.append(nn.Sequential(
                StackedConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip, num_conv_per_stage - 1,
                                  self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                                  self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
                StackedConvLayers(nfeatures_from_skip, final_num_features, 1, self.conv_op, self.conv_kwargs,
                                  self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                  self.nonlin, self.nonlin_kwargs, basic_block=basic_block)
            ))

        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(conv_op(self.conv_blocks_localization[ds][-1].output_channels, num_classes,
                                            1, 1, 0, 1, 1, seg_output_use_bias))
        self.conv_meta = conv_op(320, 2, 1, 1, 0, 1, 1, seg_output_use_bias)


        self.upscale_logits_ops = []
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        for usl in range(num_pool - 1):
            if self.upscale_logits:
                self.upscale_logits_ops.append(Upsample(scale_factor=tuple([int(i) for i in cum_upsample[usl + 1]]),
                                                        mode=upsample_mode))
            else:
                self.upscale_logits_ops.append(lambda x: x)

        if not dropout_in_localization:
            self.dropout_op_kwargs['p'] = old_dropout_p

        # register all modules properly
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.td = nn.ModuleList(self.td)
        self.tu = nn.ModuleList(self.tu)
        self.TEM_list = nn.ModuleList(self.TEM_list)
        self.RAG = nn.ModuleList(self.RAG)

        self.seg_outputs = nn.ModuleList(self.seg_outputs)

        #
        conv_kwargs_meta = {'kernel_size': 1, 'stride': 1, 'padding': 0, 'dilation': 1, 'bias': True}
        norm_op_meta = nn.InstanceNorm3d
        dropout_op_meta = nn.Dropout3d
        norm_op_kwargs_meta = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs_meta = {'p': 0, 'inplace': True}
        self.meta_conv = ConvDropoutNormNonlin(96, 320, nn.Conv3d, conv_kwargs_meta, norm_op_meta, norm_op_kwargs_meta, dropout_op_meta, dropout_op_kwargs_meta)


        self.ncd_fusion = PDC()#NCD()
        self.down = Upsample(scale_factor=[0.25, 0.125, 0.125], mode='trilinear')

        if self.upscale_logits:
            self.upscale_logits_ops = nn.ModuleList(
                self.upscale_logits_ops)  # lambda x:x is not a Module so we need to distinguish here



        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)
            # self.apply(print_module_training_status)

    def forward(self, x):
        skips = []
        seg_outputs = []
        skips_3_to_5 = []

        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            if d > 1:
                skips_3_to_5.append(self.TEM_list[d](x))  # we need change TEM-block
            if not self.convolutional_pooling:
                x = self.td[d](x)

        fusion_feat = self.ncd_fusion(skips_3_to_5[2], skips_3_to_5[1], skips_3_to_5[0])
        fusion_feat_down = self.down(fusion_feat)   # why not use fusion feature so do not need fusion
        # Downsample x [2 96 6 6 6]  # zzz step be step   first down second conv

        x = self.meta_conv(fusion_feat_down)
        # x = self.conv_blocks_context[-1](x) #[2 320 6 6 6]

        # x = self.meta_fusion(fusion_feat_down, x)

        for u in range(len(self.tu)):
            if u > 1: # ori<2
                x = self.tu[u](x)
                x = torch.cat((x, skips[-(u + 1)]), dim=1)
                x = self.conv_blocks_localization[u](x)
                seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))
            else:
                refinement_skips = self.RAG[u](skips[-(u + 1)], x)    # fiver layer
                x = self.tu[u](x)
                x = torch.cat((x, refinement_skips), dim=1)
                x = self.conv_blocks_localization[u](x)
                seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]

    @staticmethod
    def compute_approx_vram_consumption(patch_size, num_pool_per_axis, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes, deep_supervision=False,
                                        conv_per_stage=2):
        """
        This only applies for num_conv_per_stage and convolutional_upsampling=True
        not real vram consumption. just a constant term to which the vram consumption will be approx proportional
        (+ offset for parameter storage)
        :param deep_supervision:
        :param patch_size:
        :param num_pool_per_axis:
        :param base_num_features:
        :param max_num_features:
        :param num_modalities:
        :param num_classes:
        :param pool_op_kernel_sizes:
        :return:
        """
        if not isinstance(num_pool_per_axis, np.ndarray):
            num_pool_per_axis = np.array(num_pool_per_axis)

        npool = len(pool_op_kernel_sizes)

        map_size = np.array(patch_size)
        tmp = np.int64((conv_per_stage * 2 + 1) * np.prod(map_size, dtype=np.int64) * base_num_features +
                       num_modalities * np.prod(map_size, dtype=np.int64) +
                       num_classes * np.prod(map_size, dtype=np.int64))

        num_feat = base_num_features

        for p in range(npool):
            for pi in range(len(num_pool_per_axis)):
                map_size[pi] /= pool_op_kernel_sizes[p][pi]
            num_feat = min(num_feat * 2, max_num_features)
            num_blocks = (conv_per_stage * 2 + 1) if p < (npool - 1) else conv_per_stage  # conv_per_stage + conv_per_stage for the convs of encode/decode and 1 for transposed conv
            tmp += num_blocks * np.prod(map_size, dtype=np.int64) * num_feat
            if deep_supervision and p < (npool - 2):
                tmp += np.prod(map_size, dtype=np.int64) * num_classes
            # print(p, map_size, num_feat, tmp)
        return tmp


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
    network = Generic_UNet(1, 32, 2,
                 len(net_num_pool_op_kernel_sizes),
                 2, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                 dropout_op_kwargs,
                 net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                 net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True)

    data =torch.randn(2, 1, 48, 192, 192)
    # block = DownConv(32, 32, first_stride=[1, 2, 2])
    print(network)
    pred = network(data)
    print(pred[0].shape)
    print(pred[1].shape)
    print(pred[2].shape)
    print(pred[3].shape)
    print(pred[4].shape)
