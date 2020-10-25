""" This file contains the model definitions for both original ResNet (6n+2
layers) and SkipNets.
"""

import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, kernel_size=3):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn3 = nn.BatchNorm2d(planes)
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
            residual = self.bn3(residual)

        out  += residual
        out = self.relu(out)
        return out


########################################
# Original ResNet                      #
########################################

class ResNet(nn.Module):
    """Original ResNet without routing modules"""
    def __init__(self, block, layers, num_classes=10):
        self.inplanes = 16
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        self.num_layers = layers
        
        self._make_group(block, 16, layers[0], group_id=1,
                         )
        self._make_group(block, 32, layers[1], group_id=2,
                         )
        self._make_group(block, 64, layers[2], group_id=3,
                         )

        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(0) * m.weight.size(1)
                m.weight.data.normal_(0, math.sqrt(2. / n))

                
    def _make_group(self, block, planes, layers, group_id=1
                    ):
        """ Create the whole group"""
        for i in range(layers):
            if group_id > 1 and i == 0:
                stride = 2
            else:
                stride = 1

            layer = self._make_layer_v2(block, planes, stride=stride)

            # setattr(self, 'group{}_ds{}'.format(group_id, i), meta[0])
            setattr(self, 'group{}_layer{}'.format(group_id, i), layer)
            # setattr(self, 'group{}_gate{}'.format(group_id, i), meta[2])
            
            
    def _make_layer_v2(self, block, planes, stride=1):
        """ create one block and optional a gate module """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)
            
        layer = block(self.inplanes, planes, stride, downsample)
        self.inplanes = planes * block.expansion
            
        return layer
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        for g in range(3):
            for i in range(self.num_layers[g]):
                x = getattr(self, 'group{}_layer{}'.format(g+1, i))(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# For CIFAR-10
# ResNet-38

def cifar10_resnet_20(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [3, 3, 3], **kwargs)
    return model

def cifar10_resnet_31(pretrained=False, **kwargs):
    # n = 5
    model = ResNet(BasicBlock, [5, 5, 5], **kwargs)
    return model



def cifar10_resnet_38(pretrained=False, **kwargs):
    # n = 6
    model = ResNet(BasicBlock, [6, 6, 6], **kwargs)
    return model


# ResNet-74
def cifar10_resnet_74(pretrained=False, **kwargs):
    # n = 12
    model = ResNet(BasicBlock, [12, 12, 12], **kwargs)
    return model


# ResNet-110
def cifar10_resnet_110(pretrained=False, **kwargs):
    # n = 18
    model = ResNet(BasicBlock, [18, 18, 18], **kwargs)
    return model


# ResNet-152
def cifar10_resnet_152(pretrained=False, **kwargs):
    # n = 25
    model = ResNet(BasicBlock, [25, 25, 25], **kwargs)
    return model


# For CIFAR-100
# ResNet-38
def cifar100_resnet_38(pretrained=False, **kwargs):
    # n = 6
    model = ResNet(BasicBlock, [6, 6, 6], num_classes=100)
    return model


# ResNet-74
def cifar100_resnet_74(pretrained=False, **kwargs):
    # n = 12
    model = ResNet(BasicBlock, [12, 12, 12], num_classes=100)
    return model


# ResNet-110
def cifar100_resnet_110(pretrained=False, **kwargs):
    # n = 18
    model = ResNet(BasicBlock, [18, 18, 18], num_classes=100)
    return model


# ResNet-152
def cifar100_resnet_152(pretrained=False, **kwargs):
    # n = 25
    model = ResNet(BasicBlock, [25, 25, 25], num_classes=100)
    return model
