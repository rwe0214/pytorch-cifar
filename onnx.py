import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from models import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Evaluating')
parser.add_argument('--model_arch', type=str, default='resnet34', help=' model backbone type')
parser.add_argument('--model_path', type=str, help='the path of model\'s weight')
parser.add_argument('--pretrain_layer', default=1, type=int)
parser.add_argument('--onnx_batch_size', default=1, type=int)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('==> Building model..')
resnet_archs = {
            'resnet18': CloudResNet18,
            'resnet34': CloudResNet34,
            'resnet50': CloudResNet50,
            'resnet506': CloudResNet50_6,
            'densenet121': CloudDenseNet121
            }

net = resnet_archs[args.model_arch](depth = args.pretrain_layer, num_classes = 10).to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

state_dict = torch.load(args.model_path)
# for name, weights in state_dict['net'].items():
#     state_dict['net'][name] = (weights*(1<<12)).int()
# print(state_dict)
net.load_state_dict(state_dict['net'])
net.eval()
# with open('firstbatch_1_x.npy', 'rb') as f:
#     x = np.load(f)
# print((x * (1<<12)).astype(int))
# print(net((torch.from_numpy(x).to(device))))

print('==> Exporting model to onnx')
net = net.module.to('cpu')
x = torch.randn(args.onnx_batch_size, 3, 32, 32)
y = net(x)
torch.onnx.export(net,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  f'./onnx/{args.model_arch}_{args.pretrain_layer}_{args.onnx_batch_size}.onnx',   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  verbose=True,
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'],) # the model's output names
                  # dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                  #               'output' : {0 : 'batch_size'}})

#print(net(torch.randn(1,3,32,32)).shape)

