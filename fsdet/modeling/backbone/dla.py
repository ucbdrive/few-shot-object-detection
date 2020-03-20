#!/usr/bin/env python
# -*- coding: utf-8 -*-
# adapted from https://github.com/ucbdrive/dla

import math
import numpy as np
from os.path import join
import torch
from torch import nn
import torch.utils.model_zoo as model_zoo
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from .backbone import Backbone
from .build import BACKBONE_REGISTRY
from fsdet.layers import  ShapeSpec


__all__ = []


BatchNorm = nn.BatchNorm2d

WEB_ROOT = 'http://dl.yf.io/dla/models'

model_urls = {
    'dla34': join(WEB_ROOT, 'imagenet', 'dla34-ba72cf86.pth'),
    'dla46_c': join(WEB_ROOT, 'imagenet', 'dla46_c-2bfd52c3.pth'),
    'dla46x_c': join(WEB_ROOT, 'imagenet', 'dla46x_c-d761bae7.pth'),
    'dla60': join(WEB_ROOT, 'imagenet', 'dla60-24839fc4.pth'),
    'dla60x': join(WEB_ROOT, 'imagenet', 'dla60x-d15cacda.pth'),
    'dla60x_c': join(WEB_ROOT, 'imagenet', 'dla60x_c-b870c45c.pth'),
    'dla102': join(WEB_ROOT, 'imagenet', 'dla102-d94d9790.pth'),
    'dla102x': join(WEB_ROOT, 'imagenet', 'dla102x-ad62be81.pth'),
    'dla102x2': join(WEB_ROOT, 'imagenet', 'dla102x2-262837b6.pth'),
    'dla169': join(WEB_ROOT, 'imagenet', 'dla169-0914e092.pth')
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = BatchNorm(planes)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = BatchNorm(bottle_planes)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = BatchNorm(bottle_planes)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
        # bottle_planes = dim * cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = BatchNorm(bottle_planes)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation, groups=cardinality)
        self.bn2 = BatchNorm(bottle_planes)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                BatchNorm(out_channels)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(Backbone):
    def __init__(self, levels, channels, out_features, input_channels=3,
                 num_classes=1000, pool_size=7,
                 block=BasicBlock,
                 residual_root=False, scale_idx=5):
        super(DLA, self).__init__()
        self.channels = channels
        # removed the usage of num classes for similicity
        self.num_classes = num_classes
        # as part of the property defined in `Backbone`
        self._out_features = out_features
        self._out_feature_strides = {}
        self._out_feature_channels = {}

        # self.return_levels = return_levels
        self.scale_idx = scale_idx
        self.out_channels = channels[self.scale_idx]
        self.base_layer = nn.Sequential(
            nn.Conv2d(input_channels, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            BatchNorm(channels[0]),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self._out_feature_channels['level0'] = channels[0]
        self._out_feature_strides['level0'] = 1

        if self.scale_idx >= 1:
            self.level1 = self._make_conv_level(
                channels[0], channels[1], levels[1], stride=2)
            self._out_feature_channels['level1'] = channels[1]
            self._out_feature_strides['level1'] = 2

        if self.scale_idx >= 2:
            self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                               level_root=False,
                               root_residual=residual_root)
            self._out_feature_channels['level2'] = channels[2]
            self._out_feature_strides['level2'] = 4

        if self.scale_idx >= 3:
            self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                               level_root=True, root_residual=residual_root)
            self._out_feature_channels['level3'] = channels[3]
            self._out_feature_strides['level3'] = 8

        if self.scale_idx >= 4:
            self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                               level_root=True, root_residual=residual_root)
            self._out_feature_channels['level4'] = channels[4]
            self._out_feature_strides['level4'] = 16

        if self.scale_idx >= 5:
            self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                               level_root=True, root_residual=residual_root)
            self._out_feature_channels['level5'] = channels[5]
            self._out_feature_strides['level5'] = 32

        self.avgpool = nn.AvgPool2d(pool_size)
        self.fc = nn.Conv2d(channels[-1], num_classes, kernel_size=1,
                            stride=1, padding=0, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                BatchNorm(planes),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                BatchNorm(planes),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = {}
        x = self.base_layer(x)
        for i in range(self.scale_idx + 1):
            name = 'level{}'.format(i)
            x = getattr(self, name)(x)
            if name in self._out_features:
                y[name] = x
        return y

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_features_channels[name],
                stride=self._out_features_strides[name]
            )
            for name in self._out_features
        }


def dla34(**kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)

    state_dict = load_state_dict_from_url(model_urls['dla34'])
    model.load_state_dict(state_dict)
    return model


def dla46_c(**kwargs):  # DLA-46-C
    Bottleneck.expansion = 2
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 64, 128, 256],
                block=Bottleneck, **kwargs)
    state_dict = load_state_dict_from_url(model_urls['dla46_c'])
    model.load_state_dict(state_dict)

    return model


def dla46x_c(**kwargs):  # DLA-X-46-C
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 64, 128, 256],
                block=BottleneckX, **kwargs)
    state_dict = load_state_dict_from_url(model_urls['dla46x_c'])
    model.load_state_dict(state_dict)
    return model


def dla60x_c(**kwargs):  # DLA-X-60-C
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 2, 3, 1],
                [16, 32, 64, 64, 128, 256],
                block=BottleneckX, **kwargs)
    state_dict = load_state_dict_from_url(model_urls['dla60x_c'])
    model.load_state_dict(state_dict)
    return model


def dla60(**kwargs):  # DLA-60
    Bottleneck.expansion = 2
    model = DLA([1, 1, 1, 2, 3, 1],
                [16, 32, 128, 256, 512, 1024],
                block=Bottleneck, **kwargs)
    state_dict = load_state_dict_from_url(model_urls['dla60'])
    model.load_state_dict(state_dict)
    return model


def dla60x(**kwargs):  # DLA-X-60
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 2, 3, 1],
                [16, 32, 128, 256, 512, 1024],
                block=BottleneckX, **kwargs)
    state_dict = load_state_dict_from_url(model_urls['dla60x'])
    model.load_state_dict(state_dict)
    return model


def dla102(**kwargs):  # DLA-102
    Bottleneck.expansion = 2
    model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                block=Bottleneck, residual_root=True, **kwargs)
    state_dict = load_state_dict_from_url(model_urls['dla102'])
    model.load_state_dict(state_dict)
    return model


def dla102x(**kwargs):  # DLA-X-102
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                block=BottleneckX, residual_root=True, **kwargs)
    state_dict = load_state_dict_from_url(model_urls['dla102x'])
    model.load_state_dict(state_dict)
    return model


def dla102x2(**kwargs):  # DLA-X-102 64
    BottleneckX.cardinality = 64
    model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                block=BottleneckX, residual_root=True, **kwargs)
    state_dict = load_state_dict_from_url(model_urls['dla102x2'])
    model.load_state_dict(state_dict)
    return model


def dla169(**kwargs):  # DLA-169
    Bottleneck.expansion = 2
    model = DLA([1, 1, 2, 3, 5, 1], [16, 32, 128, 256, 512, 1024],
                block=Bottleneck, residual_root=True, **kwargs)
    state_dict = load_state_dict_from_url(model_urls['dla169'])
    model.load_state_dict(state_dict)
    return model


@BACKBONE_REGISTRY.register()
def build_dla_backbone(cfg, input_shape):

    # return levels for backbone with FPN
    args = {
        'input_channels': input_shape.channels,
        'out_features': cfg.MODEL.DLA.OUT_FEATURES
    }

    if cfg.MODEL.DLA.ARCH == 'DLA-34':
        return dla34(**args)
    elif cfg.MODEL.DLA.ARCH == 'DLA-46-C':
        return dla46_c(**args)
    elif cfg.MODEL.DLA.ARCH == 'DLA-X-46-C':
        return dla46x_c(**args)
    elif cfg.MODEL.DLA.ARCH == 'DLA-X-60-C':
        return dla60x_c(**args)
    elif cfg.MODEL.DLA.ARCH == 'DLA-60':
        return dla60(**args)
    elif cfg.MODEL.DLA.ARCH == 'DLA-X-60':
        return dla60x(**args)
    elif cfg.MODEL.DLA.ARCH == 'DLA-102':
        return dla102(**args)
    elif cfg.MODEL.DLA.ARCH == 'DLA-X-102':
        return dla102x(**args)
    elif cfg.MODEL.DLA.ARCH == 'DLA-X-102-2':
        return dla102x2(**args)
    elif cfg.MODEL.DLA.ARCH == 'DLA-169':
        return dla169(**args)
    else:
        raise ValueError
