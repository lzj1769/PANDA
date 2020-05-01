import torch.nn as nn
from efficientnet import EfficientNet


class PandaEfficientNet(nn.Module):
    def __init__(self, arch, num_classes=1, pretrained=True):
        super().__init__()

        if pretrained:
            self.base = EfficientNet.from_pretrained(model_name=arch)
        else:
            self.base = EfficientNet.from_name(model_name=arch)

        self.in_features = self.base._fc.in_features

        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(self.in_features, num_classes)

    def forward(self, inputs):
        bs, num_tiles, c, h, w = inputs.size()
        inputs = inputs.view(-1, c, h, w)

        # Convolution layers
        x = self.base.extract_features(inputs)
        shape = x.shape

        # concatenate the output for tiles into a single map
        x = x.view(-1, bs, shape[1], shape[2], shape[3]).permute(0, 2, 1, 3, 4).contiguous() \
            .view(-1, shape[1], shape[2] * num_tiles, shape[3])  # x: bs x C x N*4 x 4

        # Pooling and final linear layer
        x = self.avg_pooling(x)
        x = x.view(bs, -1)
        x = self.dropout(x)
        x = self.fc1(x)

        return x
