import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class ConcatLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConcatLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels[1] * 2, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.resolute = in_channels[0] != in_channels[1]
        if self.resolute:
            self.conv2 = nn.Conv2d(in_channels[0], in_channels[1], kernel_size=1, bias=False)
            self.bn2 = nn.BatchNorm2d(in_channels[1])

    def forward(self, x, y):
        if self.resolute:
            y = F.relu(self.bn2(self.conv2(y)))
        return F.relu(self.bn1(self.conv1(torch.cat((x, y), 1))))
    
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layers = nn.ModuleList()
        self.layers.append(self._make_layer(block, 64, num_blocks[0], stride=1))
        self.layers.append(self._make_layer(block, 128, num_blocks[1], stride=2))
        self.layers.append(self._make_layer(block, 256, num_blocks[2], stride=2))
        self.layers.append(self._make_layer(block, 512, num_blocks[3], stride=2))
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        for depth, layer in enumerate(self.layers):
            out = layer(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
class EdgeResNet(nn.Module):
    def __init__(self, block, num_blocks, cloud_model, depth, num_classes=10):
        '''
            depth: depth of the concat layer
        '''
        super(EdgeResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layers = nn.ModuleList()
        self.layers.append(self._make_layer(block, 64, num_blocks[0], stride=1))
        self.layers.append(self._make_layer(block, 128, num_blocks[1], stride=2))
        self.layers.append(self._make_layer(block, 256, num_blocks[2], stride=2))
        self.layers.append(self._make_layer(block, 512, num_blocks[3], stride=2))
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.depth = depth
        self.cloud_model = cloud_model

        self.expansion = cloud_model.layers[depth-1][-1].expansion.bit_length()-1
        self.concat_channels = [2 ** (depth + 5 + self.expansion), 2 ** (depth + 5)]
        self.concat_layer = ConcatLayer(self.concat_channels, self.concat_channels[1])

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, y):
        out = F.relu(self.bn1(self.conv1(x)))
        cloud_out = self.cloud_model(y)
        for depth, layer in enumerate(self.layers):
            if self.depth == depth:
                out = self.concat_layer(out, cloud_out)
            out = layer(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
class CloudResNet(nn.Module):
    def __init__(self, block, num_blocks, depth, num_classes=10):
        super(CloudResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layers = nn.ModuleList()
        self.layers.append(self._make_layer(block, 64, num_blocks[0], stride=1))
        self.layers.append(self._make_layer(block, 128, num_blocks[1], stride=2))
        self.layers.append(self._make_layer(block, 256, num_blocks[2], stride=2))
        self.layers.append(self._make_layer(block, 512, num_blocks[3], stride=2))
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.depth = depth

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        for depth, layer in enumerate(self.layers):
            if self.depth == depth:
                return out
            out = layer(out)
            
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
def ResNet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def ResNet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

def ResNet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def ResNet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

def ResNet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

def CloudResNet18(**kwargs):
    return CloudResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def CloudResNet34(**kwargs):
    return CloudResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

def CloudResNet50(**kwargs):
    return CloudResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def CloudResNet101(**kwargs):
    return CloudResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

def CloudResNet152(**kwargs):
    return CloudResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

def EdgeResNet18(**kwargs):
    return EdgeResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def EdgeResNet34(**kwargs):
    return EdgeResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

def EdgeResNet50(**kwargs):
    return EdgeResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def EdgeResNet101(**kwargs):
    return EdgeResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

def EdgeResNet152(**kwargs):
    return EdgeResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

def test():
    concat_layer = 3
    cloud_model = CloudResNet101(depth=concat_layer, num_classes=10)
    edge_model = EdgeResNet18(cloud_model=cloud_model, depth=concat_layer, num_classes=100)
    x = torch.randn(1, 3, 32, 32)
    y = torch.randn(1, 3, 32, 32)

    out = edge_model(x, y)
    print(edge_model.concat_layer)
    print(out.shape)
    
if __name__ == '__main__':
    test()
