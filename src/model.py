import torch
import torch.nn as nn

from inceptionv4 import inceptionv4
from inceptionresnetv2 import inceptionresnetv2
from senet import se_resnext50_32x4d, se_resnext101_32x4d
from mish import Mish


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        self.output_size = sz
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class PandaNet(nn.Module):
    def __init__(self, arch, num_classes=1, num_patches=None, pretrained=True):
        super().__init__()

        # load EfficientNet
        if arch == 'se_resnext50_32x4d':
            if pretrained:
                self.base = se_resnext50_32x4d()
            else:
                self.base = se_resnext50_32x4d(pretrained=None)
            self.nc = self.base.last_linear.in_features

        elif arch == 'se_resnext101_32x4d':
            if pretrained:
                self.base = se_resnext101_32x4d()
            else:
                self.base = se_resnext101_32x4d(pretrained=None)
            self.nc = self.base.last_linear.in_features

        elif arch == 'inceptionv4':
            if pretrained:
                self.base = inceptionv4()
            else:
                self.base = inceptionv4(pretrained=None)
            self.nc = self.base.last_linear.in_features

        elif arch == 'inceptionresnetv2':
            if pretrained:
                self.base = inceptionresnetv2()
            else:
                self.base = inceptionresnetv2(pretrained=None)
            self.nc = self.base.last_linear.in_features

        self.pool = nn.Sequential(AdaptiveConcatPool2d(1),
                                  Flatten(),
                                  nn.BatchNorm1d(2 * self.nc),
                                  nn.Dropout(0.5),
                                  nn.Linear(2 * self.nc, 1),
                                  Mish())

        self.fc = nn.Linear(num_patches, 1)

    def forward(self, x):
        bs, num_patches, c, h, w = x.size()

        x = self.base.features(x.view(-1, c, h, w))  # bs*N x c x h x w
        x = self.pool(x)
        x = x.view(-1, num_patches)

        # Pooling and final linear layer
        x = self.fc(x)

        return x
