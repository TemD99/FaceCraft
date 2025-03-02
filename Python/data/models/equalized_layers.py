


import sys
if '/app' not in sys.path:
    sys.path.append('/app')

import torch
from torch import nn
from math import sqrt
import torch.nn.functional as F 
import numpy as np
from Python.data.SG2.config import *


sys.path.append('/app/Python/data/SG2')

# Learning-rate Equalized Linear Layer
#   equalize the learning rate for a linear layer
class EqualizedLinear(nn.Module):

    def __init__(self, in_features, out_features, bias = 0.):
        # in_features is the number of features in the input feature map
        # out_features is the number of features in the output feature map

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # initialize the weight
        self.weight = EqualizedWeight([out_features, in_features])
        # initialize the bias
        self.bias = nn.Parameter(torch.ones(out_features) * bias)

    def forward(self, x: torch.Tensor):
        # x is the input feature map of the shape [batch_size, in_features, height, width]

        # return the linear transformation of x, weight, and bias
        return F.linear(x, self.weight(), bias=self.bias)

# Learning-rate Equalized 2D Convolution Layer
#   equalize the learning rate for a convolution layer
class EqualizedConv2d(nn.Module):

    def __init__(self, in_features, out_features,
                 kernel_size, padding = 0):
        # in_features is the number of features in the input feature map
        # out_features is the number of features in the output feature map

        super().__init__()
        # initialize the padding
        self.padding = padding
        # initialize weight by a class EqualizedWeight
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        # initialize bias
        self.bias = nn.Parameter(torch.ones(out_features))

    def forward(self, x: torch.Tensor):
        # x is the input feature map of the shape [batch_size, in_features, height, width]

        # return the convolution of x, weight, bias, and padding
        return F.conv2d(x, self.weight(), bias=self.bias, padding=self.padding)

# Learning-rate Equalized Weights Parameter
#  Instead of initializing weights at N(0,c) they initialize weights to N(0,1) and then multiply them by c when using it
class EqualizedWeight(nn.Module):

    def __init__(self, shape):
        # shape of the weight parameter
        super().__init__()
        # initialize the constant c
        self.c = 1 / sqrt(np.prod(shape[1:]))
        # initialize weights with N(0,1)
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self):
        # multiply weights by c and return
        return self.weight * self.c
