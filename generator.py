
import math
import torch
from torch import nn, optim
import torch.nn.functional as F 
import torchvision

class UpBlock(nn.Module):
    def __init__(self, in_channel, out_channel=None, factor=2, activate=True):
        super(UpBlock, self).__init__()
        self.activate = activate
        out_channel = out_channel if out_channel is not None else in_channel
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=factor, mode='linear', align_corners=True),
            nn.Conv1d(in_channel, out_channel, 3,1,2, dilation=2, bias=False),
            nn.BatchNorm1d(out_channel),
        )

    def zero_init(self):
        if self.block[1].weight is not None:
            # nn.init.normal_(self.block[1].weight, mean=0, std=0.000001)
            nn.init.constant_(self.block[1].weight, val=0)
        if self.block[1].bias is not None:
            # nn.init.normal_(self.block[1].bias, mean=0, std=0.000001)
            nn.init.constant_(self.block[1].bias, val=0)

    def forward(self, x):
        x = self.block(x)
        if self.activate:
            x = F.relu(x, inplace=True)
        return x


class InterpolationBlock(nn.Module):
    def __init__(self, in_channel,  output_dim, out_channel=None, activate=True):
        super(InterpolationBlock, self).__init__()
        self.activate = activate
        out_channel = out_channel if out_channel is not None else in_channel
        self.block = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, 3,1,2, dilation=2, bias=False),
            nn.BatchNorm1d(out_channel),
        )
        self.output_dim = output_dim

    def zero_init(self):
        if self.block[1].weight is not None:
            # nn.init.normal_(self.block[1].weight, mean=0, std=0.000001)
            nn.init.constant_(self.block[1].weight, val=0)
        if self.block[1].bias is not None:
            # nn.init.normal_(self.block[1].bias, mean=0, std=0.000001)
            nn.init.constant_(self.block[1].bias, val=0)

    def forward(self, x):
        x = F.interpolate(x, self.output_dim, mode='linear')
        x = self.block(x)
        if self.activate:
            x = F.relu(x, inplace=True)
        return x


class LinearBlock(nn.Module):
    def __init__(self, in_channel, out_channel=None):
        super(LinearBlock, self).__init__()
        out_channel = out_channel if out_channel is not None else in_channel
        self.block = nn.Sequential(
            nn.Linear(in_channel, out_channel, bias=False),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x_shape = x.shape
        B, C, F = x_shape
        return self.block(x.view(B*C, F)).view(x_shape)


class Identity(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        return x


class Generator1D(nn.Module):
    def __init__(self, noise_dim=100, output_dim=3200):
        super(Generator1D, self).__init__()
        self.noise_dim, self.output_dim = noise_dim, output_dim
        self.base_channel, self.base_size = 64, 25
        self.transform_layer = nn.Linear(noise_dim, self.base_channel*self.base_size, bias=True)
        self.blocks = self.get_blocks(self.base_channel, self.base_size, output_dim)
        self.out_layer = nn.Conv1d(self.base_channel,1,3,1,1,bias=False)
        self.last_zero_init()

    def last_zero_init(self):
        if self.out_layer.weight is not None:
            nn.init.constant_(self.out_layer.weight, val=0)
        if self.out_layer.bias is not None:
            nn.init.constant_(self.out_layer.bias, val=0)
        # self.blocks[-1].zero_init()
        
    def get_blocks(self, base_channel, base_size, output_dim):
        layers = int(math.log2(output_dim//base_size))
        blocks = []
        in_dim = base_size
        channel = base_channel
        for idx in range(layers):
            blocks.append(UpBlock(in_channel=channel, out_channel=channel, activate=True))
            in_dim *= 2
        if not in_dim == output_dim:
            blocks.append(InterpolationBlock(in_channel=channel, output_dim=output_dim, out_channel=channel, activate=True))

        return nn.Sequential(*blocks)


    def forward(self, x):
        B = x.shape[0]
        x = self.transform_layer(x)
        x= x.view(B, self.base_channel, self.base_size)
        # print(x.shape)
        x = self.out_layer(self.blocks(x))
        # print(x.shape)
        # x = self.blocks(x)
        return x.view(B, -1)
        

