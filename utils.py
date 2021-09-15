'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import copy

import torch.nn as nn
import torch.nn.init as init


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def load_keyword(net, state_dict, keyword='cloud_model'):
    model_dict = net.state_dict()
    for name, weight in state_dict.items():
        name = name.replace('module', 'module.'+keyword)
        model_dict[name] = weight
    net.load_state_dict(model_dict)

def partial_load(net, state_dict, num_layer=-1):
    if num_layer == -1:
        return

    model_dict = net.state_dict()
    excluded_layer_name = f'layer{num_layer+1}'
    for name, weight in state_dict.items():
        if excluded_layer_name in name:
            break
        model_dict[name] = weight

    net.load_state_dict(model_dict)

def freeze_layer(net, num_layer=-1):
    if num_layer == -1:
        return net

    excluded_layer_name = f'layer{num_layer+1}'
    for name, parameter in net.named_parameters():
        if excluded_layer_name in name:
            break
        parameter.requires_grad = False

    return net

def freeze_bn(net, num_layer=-1):
    if num_layer == -1:
        return net

    excluded_layer_name = f'layer{num_layer+1}'
    for name, module in net.named_modules():
        if excluded_layer_name in name:
            break
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    return net

def freeze_keyword(net, keyword='cloud_model'):
    for name, parameter in net.named_parameters():
        if keyword not in name:
            continue
        parameter.requires_grad = False
    for name, module in net.named_modules():
        if keyword not in name:
            continue
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
    return net



def different_lr(net, lr, num_layer=-1):
    if num_layer == -1:
        return net.parameters()

    params = []
    excluded_layer_name = f'layers.{num_layer}'
    excluded = False
    for name, parameter in net.named_parameters():
        if excluded_layer_name in name:
            excluded = True

        if excluded:
            params.append({'params': parameter})
        else:
            params.append({'params': parameter, 'lr': lr})

    return params

def aggregate_models(models):
    model = copy.deepcopy(models[0])
    model.cpu()
    for p in model.parameters():
        p.data.zero_()
    for m in models:
        for (p_avg, p) in zip(model.parameters(), m.parameters()):
            p_avg.data += p.data.cpu()
    for p in model.parameters():
        p.data /= len(models)
    return model

def distribute_models(avg_model, models):
    for m in models:
        for (p_avg, p) in zip(avg_model.parameters(), m.parameters()):
            p.data = p_avg.to(p.device)

import cv2
import numpy as np
import torch
class DifferentialPrivacy:
    def __init__(self, dataset, sensitivity = None):
        if sensitivity is None:
            total_img = np.zeros(np.array(dataset[0][0]).shape)
            imgs = []
            for img, _ in dataset:
                img = np.array(img).astype(np.double)
                total_img += img
                imgs.append(img)
            average_img = total_img / len(imgs)
            delta = [cv2.norm(average_img-img, normType=cv2.NORM_L1) for img in imgs]
            self.sensitivity = max(delta)/(len(imgs)-1)
        else:
            self.sensitivity = sensitivity

    def add_noise(self, img, e):
        # assume img is a tensor
        noise = np.random.laplace(0, self.sensitivity/e, img.shape)        
        dp_img = (img + torch.from_numpy(noise))
#         dp_img = np.clip(dp_img, 0, 1)
        return dp_img            

import h5py
import torchvision
class CIFAR100wFmap(torch.utils.data.Dataset):
    def __init__(self,
                 cifar100_root='../data', 
                 transform=None, 
                 train=True,
                 fmap_root='/mnt/disk1/CIFAR100_fmaps', 
                 model='resnet506', 
                 layer=1, 
                 scale=12,                   
                 bitlength=64):
        self.mode = 'train' if train else 'test'
        self.fmap_root = os.path.join(fmap_root, f'{model}_{layer}') 
        self.train = train
        self.scale = scale
        self.bitlength = bitlength
        self.layer = layer
        self.cifar100 = torchvision.datasets.CIFAR100(
                            root=cifar100_root, train=train, download=True, transform=transform)
        self.cifar100_transform = transform
        self.padding = {1:4, 2:2}
        self.crop_size = {1: 32, 2: 16}
        
        assert(self.is_label_order_equal())
        print('Labels\' order is verified')
   
    def __len__(self):
        return self.cifar100.__len__()
    
    def is_hflipped(self):
        # assume the transform method at index 0 is RandomHorizontalFlip
        return (self.cifar100_transform.transforms[0]).is_flipped()
    
    def get_loc(self):
        # assume the transform method at index 1 is RandomCrop
        return (self.cifar100_transform.transforms[1]).get_loc()    
    
    def is_label_order_equal(self):
        is_equal = False
        with h5py.File(os.path.join(self.fmap_root, f'cifar100_{self.mode}.hdf5')) as f:
            labels = f['labels']
            is_equal = np.array_equal(labels, np.array(self.cifar100.targets))
        return is_equal
        
    def __getitem__(self, idx):
        img, target = self.cifar100.__getitem__(idx)
        database = 'hf_fmaps' if self.train and self.is_hflipped() else 'fmaps'
        with h5py.File(os.path.join(self.fmap_root, f'cifar100_{self.mode}.hdf5')) as f:
            fmap = f[database][idx]
        
        # int2float
        fmap = torch.from_numpy(fmap.astype(np.single))
        fmap = fmap.reshape(2 ** (7 + self.layer), 2 ** (6 - self.layer), 2 ** (6 - self.layer))
        fmap = fmap / (1 << self.scale)
        
        if not self.train:
            return img, fmap, target

        # get crop location
        i, j = self.get_loc()

        # pad and crop
        padded = torch.nn.functional.pad(fmap, 
                                         (self.padding[self.layer], 
                                          self.padding[self.layer], 
                                          self.padding[self.layer], 
                                          self.padding[self.layer]))
        fmap = padded[..., i:i+self.crop_size[self.layer], j:j+self.crop_size[self.layer]]
        
        return img, fmap, target
