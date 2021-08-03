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
from tensorboard_logger import configure, log_value

torch.manual_seed(563)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--num_clients', default=1, type=int)
parser.add_argument('--cloud_arch', type=str, default='resnet34', help='pretrained cloud model backbone type')
parser.add_argument('--pretrain', type=str, help='pretrained cifar10 model')
parser.add_argument('--pretrain_layer', default=-1, type=int)
parser.add_argument('--pretrain_lr_multiplier', default=1e-3, type=float)
parser.add_argument('--log', action='store_true')
args = parser.parse_args()

if args.log is not None:
    path = f'logs/cifar100_{args.cloud_arch}_{args.num_clients}_{args.pretrain_layer}'
    if not os.path.exists(path):
        os.makedirs(path)

    configure(path)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=transform_train)
fed_trainsets = torch.utils.data.random_split(trainset, [50000 // args.num_clients] * args.num_clients)
trainloaders = [torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=4) for trainset in fed_trainsets]

testset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
resnet_archs = {
        'cloud': {
            'resnet18': CloudResNet18,
            'resnet34': CloudResNet34,
            'resnet50': CloudResNet50
            },
        'edge': {
            'resnet18': EdgeResNet18,
            'resnet34': EdgeResNet34,
            'resnet50': EdgeResNet50
            }
        }

cloud_net = resnet_archs['cloud'][args.cloud_arch](depth = args.pretrain_layer, num_classes = 10).to(device)

# load model and freeze the cloud model's parameters
edge_arch = 'resnet18'
nets = [freeze_keyword(resnet_archs['edge'][edge_arch](cloud_model = cloud_net,
                                                        depth = args.pretrain_layer,
                                                        num_classes = 100
                                                        ),
                        keyword = 'cloud_model'
                        ).to(device) for _ in range(args.num_clients)]
net_init = resnet_archs['edge'][edge_arch](cloud_model = cloud_net,
                                            depth = args.pretrain_layer,
                                            num_classes = 100
                                            )

if device == 'cuda':
    nets = [torch.nn.DataParallel(net) for net in nets]
    net_init = torch.nn.DataParallel(net_init)
    cudnn.benchmark = True

if args.pretrain:
    # load cloud_model only
    load_keyword(net_init, torch.load(args.pretrain)['net'], keyword = 'cloud_model')

distribute_models(net_init, nets)

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
        # TODO: Add differential privacy
        inputs, dp_inputs, targets = inputs.to(device), inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs, dp_inputs)
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
            # TODO: Add differential privacy
            inputs, dp_inputs, targets = inputs.to(device), inputs.to(device), targets.to(device)
            outputs = net(inputs, dp_inputs)
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
        ckpt_path = f'./checkpoint/cifar100_{args.cloud_arch}_{args.num_clients}_{args.pretrain_layer}.pth'
        torch.save(state, ckpt_path)
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    losses = []
    for net, trainloader, optimizer in zip(nets, trainloaders, optimizers):
        losses.append(train(epoch, net, trainloader, optimizer))

    if args.log is not None:
        log_value('train_loss', sum(losses) / len(losses), epoch)

    net_avg = aggregate_models(nets)
    distribute_models(net_avg, nets)

    test(epoch, nets[0])
    for scheduler in schedulers:
        scheduler.step()

