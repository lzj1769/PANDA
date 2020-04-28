import torch.nn as nn
from efficientnet import EfficientNet


class PandaEfficientNet(nn.Module):
    def __init__(self, arch):
        super().__init__()

        self.base = EfficientNet.from_pretrained(model_name=arch)
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.in_features = self.base._fc.in_features
        self.fc1 = nn.Linear(self.in_features, 1, bias=False)

    def forward(self, inputs):
        bs = inputs.size(0)
        # Convolution layers
        x = self.base.extract_features(inputs)

        # Pooling and final linear layer
        x = self.avg_pooling(x)
        x = x.view(bs, -1)
        x = self.dropout(x)
        x = self.fc1(x)

        return x
