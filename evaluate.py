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
from utils import DifferentialPrivacy as DP

torch.manual_seed(563)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Evaluating')
parser.add_argument('--model_arch', type=str, default='resnet34', help=' model backbone type')
parser.add_argument('--model_path', type=str, help='the path of model\'s weight')
parser.add_argument('--dp', action='store_true', help='using differential privacy')
parser.add_argument('--epsilon', nargs='+', default=[1, 2, 4, 8, float('inf')])
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
print('==> Preparing data..')
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=4)

if args.dp:
    privacy_agent = DP(testset, sensitivity = None)
else:
    privacy_agent = None

# Model
print('==> Loading model..')
resnet_archs = {
        'resnet18': ResNet18,
        'resnet34': ResNet34,
        'resnet50': ResNet50,
        'resnet506': ResNet50_6,
        'densenet121': DenseNet121        
    }

net = resnet_archs[args.model_arch](num_classes = 10).to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

state_dict = torch.load(args.model_path)
net.load_state_dict(state_dict['net'])

criterion = nn.CrossEntropyLoss()

# Evaluate
print('==> Evaluating model..')


epsilons = [float('inf')] if not args.dp else args.epsilon

net.eval()
acc_list = []
for e in epsilons:
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if e != float('inf'):
                inputs = privacy_agent.add_noise(inputs, e).float()
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    acc_list.append(100.*correct/total)

print('==> Evaluating result..')
print('Epsilon,Acc')
for e, acc in zip(epsilons, acc_list):
    print(f'{e},{acc}')
