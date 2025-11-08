import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from .backbone import ConvBlock, CSPBlock, SPPBlock
from .head import YoloHead

class YOLOv8(nn.Module):
    def __init__(self, num_classes, num_anchors=3):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBlock(3,32,3,1),
            ConvBlock(32,64,3,2),
            CSPBlock(64,64,1),
            ConvBlock(64,128,3,2),
            CSPBlock(128,128,2),
            ConvBlock(128,256,3,2),
            SPPBlock(256)
        )
        self.head = YoloHead(256*4, num_classes,num_anchors)

    def forward(self,x):
        x = self.stem(x)
        x = self.head(x)
        return x
