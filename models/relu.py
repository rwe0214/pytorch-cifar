import torch
import torch.nn.functional as F
import torch.nn as nn

coefs = {
    2: [0.563059, 0.5, 0.078047],
    4: [0.119782, 0.5, 0.147298, 0.0, -0.002015]
    }

def foo(x):
    return x

def approx_relu_d2(x:torch.Tensor):
    output = torch.zeros(x.shape, device=x.device)
    # x = torch.clamp(x, min=-6., max=6.)
    for exp, coef in enumerate(coefs[2]):
        output += coef * x ** exp
    return output

def approx_relu_d4(x:torch.Tensor):
    output = torch.zeros(x.shape, device=x.device)
    # x = torch.clamp(x, min=-6., max=6.)
    for exp, coef in enumerate(coefs[4]):        
        output += coef * x ** exp
    return output

def _approx_relu(degree=2):
    assert(degree==2 or degree==4)
    return approx_relu_d2 if degree == 2 else approx_relu_d4

def relu(approx=False, **kwarg):
    return F.relu if not approx else _approx_relu(**kwarg)

