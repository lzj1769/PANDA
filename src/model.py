import torch
import torch.nn as nn
from efficientnet import EfficientNet
from senet import se_resnext50_32x4d
from mish import Mish


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        self.output_size = sz
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class Lambda(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class PandaNet(nn.Module):
    def __init__(self, arch, num_classes=6, pretrained=True):
        super().__init__()

        # load EfficientNet
        if 'efficientnet' in arch:
            if pretrained:
                self.base = EfficientNet.from_pretrained(model_name=arch)
            else:
                self.base = EfficientNet.from_name(model_name=arch)

            self.extract_features = self.base.extract_features
            self.nc = self.base._fc.in_features

        elif arch == 'se_resnext50_32x4d':
            if pretrained:
                self.base = se_resnext50_32x4d()
            else:
                self.base = se_resnext50_32x4d(pretrained=None)
            self.nc = self.base.last_linear.in_features
            self.extract_features = self.base.features

        self.head = nn.Sequential(AdaptiveConcatPool2d(1),
                                  Flatten(),
                                  nn.Linear(2 * self.nc, 512),
                                  Mish(),
                                  nn.BatchNorm1d(512),
                                  nn.Dropout(0.5),
                                  nn.Linear(512, num_classes))

    def forward(self, inputs):
        bs, num_tiles, c, h, w = inputs.size()
        inputs = inputs.view(-1, c, h, w)  #

        x = self.extract_features(inputs)  # bs*N x c x h x w
        shape = x.shape

        # concatenate the output for tiles into a single map
        x = x.view(-1, num_tiles, shape[1], shape[2], shape[3]).permute(0, 2, 1, 3, 4).contiguous() \
            .view(-1, shape[1], shape[2] * num_tiles, shape[3])

        # Pooling and final linear layer
        x = self.head(x)

        return x
