import os
import logging

import torch
import torch.nn as nn
import torchvision as tv
import torchsummary
import time

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PoseResNet(nn.Module):

    def __init__(self, block, layers, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS

        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,
            extra.NUM_DECONV_FILTERS,
            extra.NUM_DECONV_KERNELS,
        )

        self.final_layer = nn.Conv2d(
            in_channels=extra.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)
        x = self.final_layer(x)

        return x

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            logger.info('=> init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.normal_(m.weight, std=0.001)
                    # nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)


resnet_spec = {
    18: (BasicBlock, [2, 2, 2, 2]),
    34: (BasicBlock, [3, 4, 6, 3]),
    50: (Bottleneck, [3, 4, 6, 3]),
    101: (Bottleneck, [3, 4, 23, 3]),
    152: (Bottleneck, [3, 8, 36, 3])
}


def get_pose_net(cfg, is_train, **kwargs):
    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS
    extra = cfg.MODEL.EXTRA

    print(num_layers)
    if num_layers == -1:
        backbone = tv.models.mobilenet_v3_large(pretrained=True)
        backbone = backbone.features

        dilated = DilatedEncoder(in_channels=960, out_channels=128)

        upsample = UpsampleBlock(in_channels=128, out_channels=extra.NUM_DECONV_FILTERS[-1],
                      upsample_block=3)
        final =  nn.Conv2d(
            in_channels=extra.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

        model = nn.Sequential(
            backbone, dilated, upsample, final
        )

    else:
        block_class, layers = resnet_spec[num_layers]

        model = PoseResNet(block_class, layers, cfg, **kwargs)

        if is_train and cfg.MODEL.INIT_WEIGHTS:
            model.init_weights(cfg.MODEL.PRETRAINED)

    batch_size = 2
    device = "cpu"
    x = torch.rand((batch_size, 3, 192, 256)).to(device)
    torchsummary.summary(model=model, input_size=x.shape[1:], device=str(device))

    torch.cuda.synchronize()
    t = time.time()
    y = model.forward(x)
    torch.cuda.synchronize()
    t = time.time() - t

    print(f'* input shape: {x.shape}')
    print(f'* output shape: {y.shape}')

    print(f'* forward time: {t:.4f} s with a batch size of {batch_size}')

    return model


class BasicConv2D(nn.Module):
    """
    Basic 2D Convolution with optional batch normalization and activation function
    """


    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, dilation=1,
                 with_batch_norm=False, activation='SiLU'):
        # type: (int, int, int, int, float, int,  bool, str) -> None
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: kernel size of the 2D convolution
        :param padding: zero-padding added to both sides of the input (default = 0)
        :param stride: stride of the convolution (default = 1)
            * NOTE: if `stride` is < 1, a trasnpsposed 2D convolution with stride=1/`stride`
            is used instead of a normal 2D convolution
        :param with_batch_norm: do you want use batch normalization?
        :param activation: activation function you want to add after the convolution
            * values in {'ReLU', 'LeakyReLU', 'Linear', 'Sigmoid', 'Tanh'}
        """
        super().__init__()

        self.with_batch_norm = with_batch_norm

        # 2D convolution
        if stride >= 1:
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=int(stride),
                padding=padding,
                dilation=dilation,
                bias=(not self.with_batch_norm)
            )
        else:
            self.conv = nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=int(1/stride)),
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=(not self.with_batch_norm)
                )
            )

            # self.conv = nn.ConvTranspose2d(
            #     in_channels=in_channels,
            #     out_channels=out_channels,
            #     kernel_size=kernel_size,
            #     stride=int(1 / stride),
            #     padding=padding,
            #     output_padding=padding,
            #     bias=(not self.with_batch_norm)
            # )

        # batch normalization
        if with_batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)

        # activation function
        assert activation in ['ReLU', 'LeakyReLU', 'Linear', 'Sigmoid', 'Tanh', 'SiLU']
        if activation == 'ReLU':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'Linear':
            self.activation = lambda x: x
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(inplace=True)
        elif activation == 'SiLU':
            self.activation = nn.SiLU(inplace=True)


    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        x = self.conv(x)
        if self.with_batch_norm:
            x = self.bn(x)
        x = self.activation(x)
        return x

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_block):
        super().__init__()

        self.deconv = []

        self.deconv.append(
            BasicConv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, stride=1))

        for i in range(upsample_block):
            self.deconv.append(
                BasicConv2D(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=0.5)
            )

        self.deconv.append(
            BasicConv2D(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
        )

        self.deconv = nn.Sequential(*self.deconv)

    def forward(self, x):
        return self.deconv(x)


class Residual(nn.Module):
    """
    A single residual block with 2 convolutional layers
    r = x | o = 1x1 conv(x) --> o = 3x3 conv(o) --> o = 1x1 conv(o) | o += r
    """


    def __init__(self, in_channels, kernel_size=3, padding=1, dilation=1, with_batch_norm=False, activation='SiLU'):
        # type: (int, int, int, int, bool, str) -> None
        """
        :param in_channels: number of input (and output) channels
        :param kernel_size: kernel size of the 2D convolution of `layer2`
        :param padding: zero-padding added to both sides of the input (default = 0)
        :param with_batch_norm: do you want use batch normalization?
        :param activation: activation function you want to add after the convolutions
            * values in {'ReLU', 'LeakyReLU', 'Linear', 'Sigmoid', 'Tanh'}
        """
        super().__init__()

        self.layer1 = BasicConv2D(
            in_channels=in_channels, out_channels=in_channels // 4, kernel_size=1, stride=1,
            padding=0, activation=activation, with_batch_norm=with_batch_norm)

        self.layer2 = BasicConv2D(
            in_channels=in_channels // 4, out_channels=in_channels // 4, kernel_size=kernel_size, stride=1,
            padding=padding, dilation=dilation
            ,activation=activation, with_batch_norm=with_batch_norm
        )

        self.layer3 = BasicConv2D(
            in_channels=in_channels // 4, out_channels=in_channels, kernel_size=1, stride=1,
            padding=0, activation=activation, with_batch_norm=with_batch_norm)

    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        residual = x
        y = self.layer1(x)
        y = self.layer2(y)
        y = self.layer3(y)
        return y + residual


class DilatedEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = BasicConv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, stride=1)
        self.conv2 = BasicConv2D(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1,
                                 stride=1)

        self.dilated_conv_blcok = nn.Sequential(
            Residual(in_channels=out_channels, kernel_size=3, dilation=2, padding=2),
            Residual(in_channels=out_channels, kernel_size=3, dilation=4, padding=4),
            Residual(in_channels=out_channels, kernel_size=3, dilation=8, padding=8),
            Residual(in_channels=out_channels, kernel_size=3, dilation=16, padding=16),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.dilated_conv_blcok(x)
        x = self.conv2(x)
        return x