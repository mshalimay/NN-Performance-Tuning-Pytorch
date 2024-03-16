"""
ResNet18 from scratch using PyTorch, with optional MobileNet's depthwise separable convolutions.
  Obs: this code does not build generalized blocks for all ResNets, just for ResNet18.

References:
  Resnet: https://arxiv.org/pdf/1512.03385v1.pdf
  Mobile Net: https://arxiv.org/pdf/1704.04861.pdf
"""

from typing import Type
import torch
import torch.nn as nn
from torch import Tensor

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.bn_depthwise = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn_depthwise(x)
        x = self.pointwise(x)
        return x


def make_convolution(in_channels: int, out_channels: int, kernel_size: int, stride: int, 
                     padding: int, bias: bool=False, mnet_conv:bool=False) -> nn.Conv2d:
    if  mnet_conv:
        return DepthwiseSeparableConv(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

# BasicBlock is a group of convolutional layers stacked together.
class BasicBlock(nn.Module):
    # class constructor
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, expansion: int = 1, downsample: nn.Module = None, mnet_conv:list=[False,False]):
        # call parent class constructor
        super(BasicBlock, self).__init__()

        # Multiplicative factor for the subsequent conv2d layer's output channels.
        # It is 1 for ResNet18 and ResNet34.
        self.expansion = expansion
        self.downsample = downsample

        # Resnet18: two convolutional convolutional layers of size 3x3xfilters

        # first convolution within the block
        self.conv1 = make_convolution(in_channels, out_channels, kernel_size=3, 
                                      stride=stride, padding=1, bias=False, mnet_conv=mnet_conv[0])
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # second convolution within the block
        self.conv2 = make_convolution(out_channels, out_channels*self.expansion, kernel_size=3, 
                                      stride=1, padding=1, bias=False, mnet_conv=mnet_conv[1])
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)

    def forward(self, x: Tensor) -> Tensor:
        # store original input for shortcut connection
        identity = x

        # Apply first convolution within a block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Apply second convolution within a block
        out = self.conv2(out)
        out = self.bn2(out)

        # downsample the input if conv make dimensions incompatible. 
        # See section 3.3 of ResNet paper.
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add shortcut connection
        out += identity

        # output of the ResNet layer
        out = self.relu(out)
        return  out

# ResNet contains 
#       (i) an initial convolutional layer applied to the input image
#       (ii) a sequence of `layers` that consist of `BasicBlock`s
#       (iii) a final fully connected layer.

# @mnet_conv: list=[False,False,False,False] is a list of boolean values that indicate whether to use depthwise separable convolution
class ResNet(nn.Module):
    """ ResNet18 from scratch using PyTorch, with optional MobileNet's depthwise separable convolutions."""
    
    def __init__(self, img_channels: int, num_layers: int, block: Type[BasicBlock], 
                 num_classes: int  = 1000, mnet_conv:list=[False,False,False,False]):
        """ ResNet18 from scratch using PyTorch, with optional MobileNet's depthwise separable convolutions.

        Args:
            img_channels (int): num of channels of the input image
            num_layers (int): num of ResNet layers (18 for ResNet18)
            block (Type[BasicBlock]): BasicBlock class
            num_classes (int, optional): num of classes of the output. Defaults to 1000.
            mnet_conv (list, optional): list of boolean values that indicate whether to use depthwise separable convolution. 
                Defaults to [False,False,False,False].
                Options:
                    0: [True, True, True, True],     # depthwise in all convolutions
                    1: [True, False, True, True],    # NOT depthwise in downsampling layer
                    2: [False, True, True, True],    # NOT depthwise in resnet initial layer
                    3: [False, False, True, True],   # NOT depthwise in resnet initial layer and downsampling
                    4: [False, False, False, True],  # depthwise only in 2nd convolution of basic block
                    5: [False, False, True, False],  # depthwise only in 1st convolution of basic block
        """

        super(ResNet, self).__init__()
        self.mnet_conv = mnet_conv
        
        if num_layers == 18:
            # `blocks_within_layer` specify the number of `BasicBlock`s within a ResNet layer.
            # eg: ResNet18 has 2 `BasicBlock`s in each of its 4 layers.
            # eg: ResNet34 has [3,4,6,3] `BasicBlock`s for each of its 4 layers.
            blocks_within_layer = [2, 2, 2, 2]
            self.expansion = 1

        # ResNet initial layer: Conv2d (7x7x64, stride 2)=> BN => ReLU => MaxPool (3x3, stride 2)
        self.in_channels = 64
        self.conv1 = make_convolution(img_channels, self.in_channels, 
                                      kernel_size=7, stride=2, padding=3, bias=False, mnet_conv=mnet_conv[0])
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers of basic blocks
        self.layer1 = self._make_layer(block, 64, blocks_within_layer[0], stride=1)
        self.layer2 = self._make_layer(block, 128, blocks_within_layer[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_within_layer[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_within_layer[3], stride=2)

        # ResNet final layers: AvgPool => FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*self.expansion, num_classes)

    def _make_layer(self, block: Type[BasicBlock], out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:

        """ eg: Resnet18 to understand logic of basic blocks, downsampling, shortcut connections (see figure 3 of ResNet paper).
          Third layer: ("conv3_x" in the paper)
            first block: output_layer2 => conv(3x3, 128, /2) => conv(3x3, 128) => x' + downsample(output_layer2)
            second block: x' + downsample(output_layer2) => conv(3x3, 128) => conv(3x3, 128) => output_layer3
          Fourth layer: ("conv4_x")
            first block: output_layer3 => conv(3x3, 256, /2) => conv(3x3, 256) => x' + downsample(output_layer3) + ...        

        Below adds a convolutional (downsample) layer to the end of a BasicBlock
        if the BasicBlock's output and the dimensions of the input are not compatible.
        This is needed to do the summation in the shortcut connection.
        Notice this downsampling is applied only one time (after the output of th FIRST block)
        See section 3.3 of ResNet paper, option B and dashed arrows of fig 3."""

        downsample = None
        if stride != 1 or self.in_channels != out_channels*self.expansion: 
            downsample = nn.Sequential(
                make_convolution(self.in_channels, out_channels*self.expansion, 
                                 kernel_size=1, stride=stride, padding=0, bias=False, mnet_conv=self.mnet_conv[1]),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

        # Add the first basic block. If `downsample` is not None, it will be added to the start of the block.
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, self.expansion, downsample, mnet_conv=self.mnet_conv[2:4]))
        
        # Stack the remaining BasicBlocks
        self.in_channels = out_channels * self.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, expansion=self.expansion, mnet_conv=self.mnet_conv[2:4]))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
