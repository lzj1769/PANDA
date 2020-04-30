import torch.nn as nn
from efficientnet import EfficientNet


class PandaEfficientNet(nn.Module):
    def __init__(self, arch, num_classes=1, pretrained=True):
        super().__init__()

        if pretrained:
            self.base = EfficientNet.from_pretrained(model_name=arch)
        else:
            self.base = EfficientNet.from_name(model_name=arch)

        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.in_features = self.base._fc.in_features
        self.fc1 = nn.Linear(self.in_features, num_classes, bias=False)
        self.fc2 = nn.Linear(self.in_features, num_classes, bias=False)

    def forward(self, inputs):
        bs = inputs.size(0)
        # Convolution layers
        x = self.base.extract_features(inputs)

        # Pooling and final linear layer
        x = self.avg_pooling(x)
        x = x.view(bs, -1)
        x = self.dropout(x)
        x1 = self.fc1(x)
        x2 = self.fc2(x)

        return x1, x2
