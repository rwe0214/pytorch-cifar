'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import zip_longest

from .relu import relu

class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate, activate_func=F.relu):
        super(Bottleneck, self).__init__()
        self.relu = activate_func

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes, activate_func=F.relu):
        super(Transition, self).__init__()
        # self.use_approx_relu = use_approx_relu
        self.relu = activate_func
        
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10, use_approx_relu=False):
        super(DenseNet, self).__init__()
        # self.use_approx_relu = use_approx_relu
        self.relu = relu(use_approx_relu)
        
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.denses = nn.ModuleList()
        self.transes = nn.ModuleList()
        for nblock in nblocks:
            if nblock == nblocks[-1]:
                self.denses.append(self._make_dense_layers(block, num_planes, nblock, self.relu))
                num_planes += nblock*growth_rate
                break
            self.denses.append(self._make_dense_layers(block, num_planes, nblock, self.relu))
            num_planes += nblock*growth_rate
            out_planes = int(math.floor(num_planes*reduction))
            self.transes.append(Transition(num_planes, out_planes, self.relu))
            num_planes = out_planes        

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock, activate_func=F.relu):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate, activate_func))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        for depth, (trans, dense) in enumerate(zip_longest(self.transes, self.denses)):
            out = trans(dense(out)) if trans is not None else dense(out)

        out = F.avg_pool2d(self.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class CloudDenseNet(nn.Module):
    def __init__(self, block, nblocks, depth, growth_rate=12, reduction=0.5, num_classes=10, use_approx_relu=False):
        super(CloudDenseNet, self).__init__()
        # self.use_approx_relu = use_approx_relu
        self.relu = relu(use_approx_relu)
        
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.denses = nn.ModuleList()
        self.transes = nn.ModuleList()
        for nblock in nblocks:
            if nblock == nblocks[-1]:
                self.denses.append(self._make_dense_layers(block, num_planes, nblock, self.relu))
                num_planes += nblock*growth_rate
                break
            self.denses.append(self._make_dense_layers(block, num_planes, nblock, self.relu))
            num_planes += nblock*growth_rate
            out_planes = int(math.floor(num_planes*reduction))
            self.transes.append(Transition(num_planes, out_planes, self.relu))
            num_planes = out_planes        

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)
        self.depth = depth
        self.backbone = 'DenseNet'

    def _make_dense_layers(self, block, in_planes, nblock, activate_func=F.relu):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate, activate_func))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        for depth, (trans, dense) in enumerate(zip_longest(self.transes, self.denses)):
            if depth == self.depth-1:
                return dense(out)
            out = trans(dense(out)) if trans is not None else dense(out)

        out = F.avg_pool2d(self.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def CloudDenseNet121(**kwargs):
    return CloudDenseNet(Bottleneck, [6,12,24,16], growth_rate=32, **kwargs)

def DenseNet121(**kwargs):
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32, **kwargs)

def DenseNet169():
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)

def DenseNet201():
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)

def DenseNet161():
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)

def densenet_cifar():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=12)

def test():
    net = CloudDenseNet121(depth=4, num_classes=10, use_approx_relu=True)
    net1 = CloudDenseNet121(depth=4, num_classes=10, use_approx_relu=False)
    x = torch.randn(1,3,32,32)
    y = net(x)
    y1 = net1(x)
    print(y.shape)
    print(torch.sub(y1, y).mean())

if __name__ == '__main__':
    test()
