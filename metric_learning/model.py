import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from efficientnet import EfficientNet
from senet import se_resnext50_32x4d, se_resnext101_32x4d


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
    def __init__(self, arch, pretrained=True, embedding_size=512):
        super().__init__()

        # load EfficientNet
        if 'efficientnet' in arch:
            if pretrained:
                self.base = EfficientNet.from_pretrained(model_name=arch)
            else:
                self.base = EfficientNet.from_name(model_name=arch)

            self.nc = self.base._fc.in_features
            self.extract_features = self.base.extract_features

        elif arch == 'se_resnext50_32x4d':
            if pretrained:
                self.base = se_resnext50_32x4d()
            else:
                self.base = se_resnext50_32x4d(pretrained=None)
            self.nc = self.base.last_linear.in_features
            self.extract_features = self.base.features

        elif arch == 'se_resnext101_32x4d':
            if pretrained:
                self.base = se_resnext101_32x4d()
            else:
                self.base = se_resnext101_32x4d(pretrained=None)
            self.nc = self.base.last_linear.in_features
            self.extract_features = self.base.features

        self.output = nn.Sequential(AdaptiveConcatPool2d(1),
                                    Flatten(),
                                    nn.BatchNorm1d(2 * self.nc),
                                    nn.Dropout(0.5),
                                    nn.Linear(2 * self.nc, 512))

    def forward(self, inputs):
        bs, num_tiles, c, h, w = inputs.size()
        inputs = inputs.view(-1, c, h, w)

        x = self.extract_features(inputs)  # bs*N x c x h x w
        shape = x.shape

        # concatenate the output for tiles into a single map
        x = x.view(-1, num_tiles, shape[1], shape[2], shape[3]).permute(0, 2, 1, 3, 4).contiguous() \
            .view(-1, shape[1], shape[2] * num_tiles, shape[3])

        # Pooling and final linear layer
        x = self.output(x)

        return x


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                    (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output
