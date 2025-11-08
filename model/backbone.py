import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c,out_c,k,s,padding=k//2,bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.LeakyReLU(0.1)
    def forward(self,x):
        return self.act(self.bn(self.conv(x)))

class CSPBlock(nn.Module):
    def __init__(self, in_c, out_c, n=1):
        super().__init__()
        c_ = out_c//2
        self.part1 = nn.Sequential(*[ConvBlock(c_,c_) for _ in range(n)])
        self.part2 = nn.Identity()
        self.merge = ConvBlock(in_c,out_c,1)
    def forward(self,x):
        x1,x2 = torch.chunk(x,2,dim=1)
        return self.merge(torch.cat([self.part1(x1),self.part2(x2)],dim=1))

class SPPBlock(nn.Module):
    def __init__(self,in_c):
        super().__init__()
        self.pool1 = nn.MaxPool2d(5,1,2)
        self.pool2 = nn.MaxPool2d(9,1,4)
        self.pool3 = nn.MaxPool2d(13,1,6)
    def forward(self,x):
        return torch.cat([x,self.pool1(x),self.pool2(x),self.pool3(x)],dim=1)
