import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from inceptionv4 import inceptionv4
from inceptionresnetv2 import inceptionresnetv2
from senet import se_resnext50_32x4d, se_resnext101_32x4d


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, output_size=None):
        super().__init__()
        self.output_size = output_size
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))


class PandaNet(nn.Module):
    def __init__(self, arch, pretrained=True):
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

        self.logit = nn.Sequential(GeM(),
                                   Flatten(),
                                   nn.BatchNorm1d(self.nc),
                                   nn.Dropout(0.2),
                                   nn.Linear(self.nc, 512),
                                   Mish(),
                                   nn.BatchNorm1d(512),
                                   nn.Dropout(0.2),
                                   nn.Linear(512, 1))

    def forward(self, x):
        bs, num_patches, c, h, w = x.size()

        x = self.base.features(x.view(-1, c, h, w))  # bs*N x c x h x w
        shape = x.shape

        x = x.view(-1, num_patches, shape[1], shape[2], shape[3])  # bs x N x c x h x w

        # concatenate the output for tiles into a single map
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(-1, shape[1], shape[2] * num_patches, shape[3])

        # Pooling and final linear layer
        x = self.logit(x)

        return x
