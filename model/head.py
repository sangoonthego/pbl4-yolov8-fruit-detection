import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch.nn as nn

class YoloHead(nn.Module):
    def __init__(self, in_c, num_classes, num_anchors=3):
        super().__init__()
        self.conv = nn.Conv2d(in_c,num_anchors*(num_classes+5),1)
    def forward(self,x):
        return self.conv(x)
