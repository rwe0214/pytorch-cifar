'''Train CIFAR100 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar, partial_load, different_lr, aggregate_models, distribute_models, freeze_keyword, load_keyword
from utils import CIFAR100wFmap
from tensorboard_logger import configure, log_value
from CachedTransforms import *

torch.manual_seed(563)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--num_clients', default=1, type=int)
parser.add_argument('--cloud_arch', type=str, default='resnet34', help='pretrained cloud model backbone type')
parser.add_argument('--pretrain', type=str, help='pretrained cifar10 model')
parser.add_argument('--pretrain_layer', default=-1, type=int)
parser.add_argument('--naive', action='store_true', help='using naive FSL')
parser.add_argument('--ezpc', action='store_true', help='using EzPC protocal')
parser.add_argument('--cl', action='store_true', help='using Conventional Learning framwork')
parser.add_argument('--fl', action='store_true', help='using Fedrated Learning framwork')
parser.add_argument('--log', action='store_true')
args = parser.parse_args()

if args.log:
    if args.ezpc:
        path = f'fsl_setting4_logs/cifar100_{args.cloud_arch}_{args.num_clients}_{args.pretrain_layer}_ezpc_replicate'
        print('Not support')
        exit()
    elif args.cl:
        path = f'fsl_setting4_logs/cifar100_{args.num_clients}N_1_cl'
    elif args.fl:
        path = f'fsl_setting4_logs/cifar100_{args.num_clients}N_{args.num_clients}_fl'
    elif args.naive:        
        path = f'fsl_setting4_logs/cifar100_{args.num_clients}N_{args.num_clients}_{args.cloud_arch}_{args.pretrain_layer}_fsl_naive'
    else:        
        path = f'fsl_setting4_logs/cifar100_{args.num_clients}N_{args.num_clients}_{args.cloud_arch}_{args.pretrain_layer}_fsl_v1'
    if not os.path.exists(path):
        os.makedirs(path)

    configure(path)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
total_clients = 16

# assert(total_clients % args.num_clients == 0)

# Data
print('==> Preparing data..')
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


trainset = torchvision.datasets.CIFAR100(root='./data', 
                                         train=True, 
                                         download=True, 
                                         transform=transform_train) if not args.ezpc else \
           CIFAR100wFmap(cifar100_root='./data', 
                         transform=transform_train, 
                         train=True,
                         fmap_root='/mnt/disk1/CIFAR100_full_fmaps', 
                         model=args.cloud_arch, 
                         layer=args.pretrain_layer, 
                         scale=12,                   
                         bitlength=64)

'''
public_trainset : private_trainsets
    * Setting1: 0 : 1
    * Setting2: 1 : 4
    * Setting3: 6 : 19
    * Setting4: 1 : 4 (public)
'''
split = [10000, 40000]
assert(len(split) == 2)

public_trainsets, private_trainset = torch.utils.data.random_split(trainset, split)
# public_trainsets = torch.utils.data.random_split(public_trainsets, [len(public_trainsets) // args.num_clients] * args.num_clients)
# assert(len(public_trainsets) == args.num_clients)

fed_trainsets = torch.utils.data.random_split(private_trainset, [len(private_trainset) // total_clients] * total_clients)
fed_trainsets = fed_trainsets[:args.num_clients]

if not args.cl:
    fed_trainsets = [torch.utils.data.ConcatDataset([public_trainsets, private]) for private in fed_trainsets]
    assert(len(fed_trainsets) == args.num_clients)
else:
    fed_trainsets.append(public_trainsets)
    fed_trainsets = [torch.utils.data.ConcatDataset(fed_trainsets)]

trainloaders = [torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=8) for trainset in fed_trainsets]


testset = torchvision.datasets.CIFAR100(root='./data', 
                                        train=False, 
                                        download=True, 
                                        transform=transform_test) if not args.ezpc else \
          CIFAR100wFmap(cifar100_root='./data', 
                        transform=transform_test, 
                        train=False,
                        fmap_root='/mnt/disk1/CIFAR100_full_fmaps', 
                        model=args.cloud_arch, 
                        layer=args.pretrain_layer, 
                        scale=12,                   
                        bitlength=64)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=8)

# Model
print('==> Building model..')
if not args.cl:
    resnet_archs = {
            'cloud': {
                'resnet18': CloudResNet18,
                'resnet34': CloudResNet34,
                'resnet50': CloudResNet50,
                'resnet506': CloudResNet50_6,
                'densenet121': CloudDenseNet121
                },
            'edge': {
                'resnet18_naive': EdgeResNet18,
                'resnet18_v1': EdgeResNet18V2,
                'resnet34': EdgeResNet34,
                'resnet50': EdgeResNet50
                }
            }
    if not args.fl:
        cloud_net = resnet_archs['cloud'][args.cloud_arch](depth = args.pretrain_layer, num_classes = 10).to(device)

        # load model and freeze the cloud model's parameters
        edge_arch = 'resnet18_v1' if not args.naive else 'resnet18_naive'
        nets = [freeze_keyword(resnet_archs['edge'][edge_arch](cloud_model = cloud_net,
                                                            depth = args.pretrain_layer,
                                                            num_classes = 100,
                                                            use_ezpc = args.ezpc
                                                            ),
                                keyword = 'cloud_model'
                                ).to(device) for _ in range(args.num_clients)]
        net_init = resnet_archs['edge'][edge_arch](cloud_model = cloud_net,
                                                    depth = args.pretrain_layer,
                                                    num_classes = 100,
                                                    use_ezpc = args.ezpc
                                                    )
    else:
        nets = [ResNet18(num_classes=100).to(device) for _ in range(args.num_clients)]
        net_init = ResNet18(num_classes=100)

    if device == 'cuda':
        nets = [torch.nn.DataParallel(net) for net in nets]
        net_init = torch.nn.DataParallel(net_init)
        cudnn.benchmark = True

    if not args.fl and args.pretrain:
        # load cloud_model only
        load_keyword(net_init, torch.load(args.pretrain)['net'], keyword = 'cloud_model')

    distribute_models(net_init, nets)

else:
    nets = [torch.nn.DataParallel(ResNet18(num_classes = 100).to(device))]
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
# train the parameters only in edge model
optimizers = [optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                        lr=args.lr,
                        momentum=0.9,
                        weight_decay=5e-4) for net in nets]
schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
              for optimizer in optimizers]

# Training
def train(epoch, net, trainloader, optimizer):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        '''if args.ezpc:
            inputs, inputs2, targets = inputs.to(device), \
                                     inputs2.to(device), \
                                     targets.to(device)
        else:
            inputs, inputs2, targets = inputs.to(device), inputs.to(device), targets.to(device)'''
        inputs, inputs2, targets = inputs.to(device), inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs) if not args.naive else net(inputs, inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return train_loss/(batch_idx+1)


def test(epoch, net):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            '''if args.ezpc:
                inputs, inputs2, targets = inputs.to(device), \
                                         inputs2.to(device), \
                                         targets.to(device)
            else:
                inputs, inputs2, targets = inputs.to(device), inputs.to(device), targets.to(device)'''
            inputs, inputs2, targets = inputs.to(device), inputs.to(device), targets.to(device)

            outputs = net(inputs) if not args.naive else net(inputs, inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    if args.log is not None:
        log_value('test_correct', 100.*correct/total, epoch)

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if args.ezpc:
            ckpt_path = f'./checkpoint/cifar100_{args.cloud_arch}_{args.num_clients}_{args.pretrain_layer}_e_{args.epsilon:.1f}.pth'
        elif args.cl:
            ckpt_path = f'./checkpoint/cifar100_{args.num_clients}N_1_cl.pth'
        elif args.fl:
            ckpt_path = f'./checkpoint/cifar100_{args.num_clients}N_{args.num_clients}_fl.pth'
        elif args.naive:
            ckpt_path = f'./checkpoint/cifar100_{args.num_clients}N_{args.num_clients}_{args.cloud_arch}_{args.pretrain_layer}_fsl_naive.pth'
        else:
            ckpt_path = f'./checkpoint/cifar100_{args.num_clients}N_{args.num_clients}_{args.cloud_arch}_{args.pretrain_layer}_fsl_v1.pth'

        torch.save(state, ckpt_path)
        best_acc = acc

for epoch in range(start_epoch, start_epoch+200):
    losses = []
    for net, trainloader, optimizer in zip(nets, trainloaders, optimizers):
        losses.append(train(epoch, net, trainloader, optimizer))

    if args.log:
        log_value('train_loss', sum(losses) / len(losses), epoch)

    if not args.cl:
        net_avg = aggregate_models(nets)
        distribute_models(net_avg, nets)

    test(epoch, nets[0])
    for scheduler in schedulers:
        scheduler.step()

