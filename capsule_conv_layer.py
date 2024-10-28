import torch.nn as nn

class CapsuleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CapsuleConvLayer, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=5, # fixme constant
                               stride=1,
                               bias=True)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.conv0(x))
