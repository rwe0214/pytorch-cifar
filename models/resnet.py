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
    
class ResNetV2(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, concat_layer=1):
        super(ResNetV2, self).__init__()
        assert(concat_layer > 0 and concat_layer < 4)
        self.in_planes = 2**(7+concat_layer)

        '''self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)'''
        self.layers = nn.ModuleList()
        # self.layers.append(self._make_layer(block, 64, num_blocks[0], stride=1))
        if concat_layer < 2:
            self.layers.append(self._make_layer(block, 128, num_blocks[1], stride=2))
        if concat_layer < 3:
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
        # out = F.relu(self.bn1(self.conv1(x)))
        out = x
        for depth, layer in enumerate(self.layers):
            out = layer(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
class ResNetV3(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, concat_layer=1):
        super(ResNetV3, self).__init__()
        assert(concat_layer > 0 and concat_layer < 4)
        self.in_planes = 2**(5+concat_layer)

        '''self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)'''
        self.layers = nn.ModuleList()
        # self.layers.append(self._make_layer(block, 64, num_blocks[0], stride=1))
        if concat_layer < 2:
            self.layers.append(self._make_layer(block, 128, num_blocks[1], stride=2))
        if concat_layer < 3:
            self.layers.append(self._make_layer(block, 256, num_blocks[2], stride=2))
        self.layers.append(self._make_layer(block, 512, num_blocks[3], stride=2))
        self.linear = nn.Linear(512*block.expansion, num_classes)
        
        self.conv1 = nn.Conv2d(2**(7+concat_layer), 2**(5+concat_layer), kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(2**(5+concat_layer))


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
    def __init__(self, block, num_blocks, cloud_model, depth, num_classes=10, use_ezpc=False):
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

        self.expansion = cloud_model.layers[depth-1][-1].expansion.bit_length()-1 if cloud_model.backbone == 'ResNet' else 2
        if cloud_model.backbone == 'DenseNet':
            # only tested on DenseNet121
            assert(self.expansion == 2)
        self.concat_channels = [2 ** (depth + 5 + self.expansion), 2 ** (depth + 5)]
        self.concat_layer = ConcatLayer(self.concat_channels, self.concat_channels[1])
        self.use_ezpc = use_ezpc

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, y):
        out = F.relu(self.bn1(self.conv1(x)))
        cloud_out = self.cloud_model(y) if not self.use_ezpc else y
        for depth, layer in enumerate(self.layers):
            if self.depth == depth:
                out = self.concat_layer(out, cloud_out)
            out = layer(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class EdgeResNetV2(nn.Module):
    def __init__(self, block, num_blocks, cloud_model, depth, num_classes=10, use_ezpc=False):
        '''
            depth: depth of the concat layer
        '''
        super(EdgeResNetV2, self).__init__()
        assert(depth > 0 and depth < 4)
        self.in_planes = 2**(7+depth)

        self.layers = nn.ModuleList()
        if depth < 2:
            self.layers.append(self._make_layer(block, 128, num_blocks[1], stride=2))
        if depth < 3:
            self.layers.append(self._make_layer(block, 256, num_blocks[2], stride=2))
        self.layers.append(self._make_layer(block, 512, num_blocks[3], stride=2))
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.depth = depth
        self.cloud_model = cloud_model

        self.expansion = cloud_model.layers[depth-1][-1].expansion.bit_length()-1 if cloud_model.backbone == 'ResNet' else 2
        if cloud_model.backbone == 'DenseNet':
            # only tested on DenseNet121
            assert(self.expansion == 2)
        self.concat_channels = [2 ** (depth + 5 + self.expansion), 2 ** (depth + 5)]
        # self.concat_layer = ConcatLayer(self.concat_channels, self.concat_channels[1])
        self.use_ezpc = use_ezpc

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.cloud_model(x) if not self.use_ezpc else x
        for depth, layer in enumerate(self.layers):
            out = layer(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class EdgeResNetV3(nn.Module):
    def __init__(self, block, num_blocks, cloud_model, depth, num_classes=10, use_ezpc=False):
        '''
            depth: depth of the concat layer
        '''
        super(EdgeResNetV3, self).__init__()
        assert(depth > 0 and depth < 4)
        self.in_planes = 2**(5+depth)

        self.layers = nn.ModuleList()
        if depth < 2:
            self.layers.append(self._make_layer(block, 128, num_blocks[1], stride=2))
        if depth < 3:
            self.layers.append(self._make_layer(block, 256, num_blocks[2], stride=2))
        self.layers.append(self._make_layer(block, 512, num_blocks[3], stride=2))
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.depth = depth
        self.cloud_model = cloud_model

        self.expansion = cloud_model.layers[depth-1][-1].expansion.bit_length()-1 if cloud_model.backbone == 'ResNet' else 2
        if cloud_model.backbone == 'DenseNet':
            # only tested on DenseNet121
            assert(self.expansion == 2)
        self.concat_channels = [2 ** (depth + 5 + self.expansion), 2 ** (depth + 5)]
        # self.concat_layer = ConcatLayer(self.concat_channels, self.concat_channels[1])
        self.use_ezpc = use_ezpc

        self.conv1 = nn.Conv2d(2**(7+depth), 2**(5+depth), kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(2**(5+depth))
        

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.cloud_model(x) if not self.use_ezpc else x
        out = F.relu(self.bn1(self.conv1(out)))
        for depth, layer in enumerate(self.layers):
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
        self.backbone = 'ResNet'

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

def ResNet18V2(**kwargs):
    return ResNetV2(BasicBlock, [2, 2, 2, 2], **kwargs)

def ResNet18V3(**kwargs):
    return ResNetV3(BasicBlock, [2, 2, 2, 2], **kwargs)

def ResNet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

def ResNet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def ResNet50_6(**kwargs):
    return ResNet(Bottleneck, [6, 3, 5, 2], **kwargs)

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

def CloudResNet50_6(**kwargs):
    return CloudResNet(Bottleneck, [6, 3, 5, 2], **kwargs)

def CloudResNet101(**kwargs):
    return CloudResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

def CloudResNet152(**kwargs):
    return CloudResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

def EdgeResNet18(**kwargs):
    return EdgeResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def EdgeResNet18V2(**kwargs):
    return EdgeResNetV2(BasicBlock, [2, 2, 2, 2], **kwargs)

def EdgeResNet18V3(**kwargs):
    return EdgeResNetV3(BasicBlock, [2, 2, 2, 2], **kwargs)

def EdgeResNet34(**kwargs):
    return EdgeResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

def EdgeResNet34V2(**kwargs):
    return EdgeResNetV2(BasicBlock, [3, 4, 6, 3], **kwargs)

def EdgeResNet50(**kwargs):
    return EdgeResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def EdgeResNet50V2(**kwargs):
    return EdgeResNetV2(Bottleneck, [3, 4, 6, 3], **kwargs)

def EdgeResNet101(**kwargs):
    return EdgeResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

def EdgeResNet152(**kwargs):
    return EdgeResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

def test():
    from thop import profile
    from thop import clever_format
    from ptflops import get_model_complexity_info
    edge_arch = {
            'v1': EdgeResNet34,
            'v2': EdgeResNet34V2}

    def prepare_input(layer):
        layer, version = layer
        x = torch.FloatTensor(1, 3, 32, 32)
        y = torch.FloatTensor(1, 2**(7+layer), 2**(6-layer), 2**(6-layer))
        return dict(x=y.to('cuda')) if version == 'v2' else dict(x=x.to('cuda'), y=y.to('cuda'))
    rows = []
    for concat_layer in range(1, 4):
        for use_ezpc in [True, False]:
            for version in ['v2']:
                if not use_ezpc:
                    continue
                cloud_model = CloudResNet50(depth=concat_layer, num_classes=10)
                edge_model = edge_arch[version](cloud_model=cloud_model, depth=concat_layer, num_classes=100, use_ezpc=use_ezpc)
                x = torch.randn(1, 3, 32, 32) if not use_ezpc else \
                    torch.randn(1, 2**(7+concat_layer), 2**(6-concat_layer), 2**(6-concat_layer))
                # out = edge_model(x)                
                macs, params = profile(edge_model, inputs=(x,)) if version == 'v2' else profile(edge_model, inputs=(torch.randn(1, 3, 32, 32), x))
                macs, params = clever_format([macs, params], "%.3f")

                args = (concat_layer, version,)
                flops, params = get_model_complexity_info(edge_model.to('cuda'), args,
                                                          input_constructor=prepare_input,
                                                          as_strings=True,
                                                          print_per_layer_stat=False)
                row = []
                row.append(f'ResNet18{version}_{concat_layer}')
                row.append(macs)
                row.append(flops)
                row.append(params)
                rows.append(row)
                print(f'ResNet{version}_{concat_layer}, EzPC({use_ezpc})')
                print(f'macs: {macs}, param: {params}')
                print(f'flops: {flops}')
    macs, params = profile(ResNet34(num_classes=100), inputs=(torch.randn(1, 3, 32, 32), ))
    macs, params = clever_format([macs, params], "%.3f")
    flops, params = get_model_complexity_info(ResNet34(num_classes=100).to('cuda'), (3,32,32),
                                              as_strings=True,
                                              print_per_layer_stat=False)
    row = []
    row.append(f'ResNet34')
    row.append(macs)
    row.append(flops)
    row.append(params)
    rows.append(row)
    import prettytable as pt
    tb = pt.PrettyTable()
    tb.field_names = ['Network', 'MACs', 'FLOPs', 'Params']
    for row in rows:
        tb.add_row(row)
    tb.sortby = 'Network'
    print(tb)
   
def test_latency():
    import time
    layer = 2
    x = torch.FloatTensor(1, 2**(7+layer), 2**(6-layer), 2**(6-layer))
    cloud_model = CloudResNet50(depth=layer, num_classes=10)
    edge_model = EdgeResNet18V2(cloud_model=cloud_model, depth=layer, num_classes=100, use_ezpc=True)
    start_time = time.time()
    y = edge_model(x)
    end_time = time.time()
    print(end_time-start_time)

    start_time = time.time()
    y = cloud_model(torch.FloatTensor(1, 3, 32, 32))
    end_time = time.time()
    print(end_time-start_time)




if __name__ == '__main__':
    test_latency()
