import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import nn, Tensor
import torchvision.models as models
from asdfghjkl.operations import Bias, Scale

#from asdfghjkl.operations.conv_aug import Conv2dAug

def get_activation(act_str):
    if act_str == 'relu':
        return nn.ReLU
    elif act_str == 'tanh':
        return nn.Tanh
    elif act_str == 'selu':
        return nn.SELU
    elif act_str == 'silu':
        return nn.SiLU
    else:
        raise ValueError('invalid activation')
    
class PrunedLeNet(nn.Sequential):
    def __init__(self, n_filters_conv1, n_filters_conv2, n_filters_conv3, 
                 in_channels=1, n_out=10, activation='relu', n_pixels=28, augmented=False):
        super().__init__()
        mid_kernel_size = 3 if n_pixels == 28 else 5
        act = get_activation(activation)
        conv = nn.Conv2d
        pool = nn.MaxPool2d
        flatten = nn.Flatten(start_dim=2) if augmented else nn.Flatten(start_dim=1)

        self.add_module('conv1', conv(in_channels, n_filters_conv1, 5, 1))
        self.add_module('act1', act())
        self.add_module('pool1', pool(2))
        self.add_module('conv2', conv(n_filters_conv1, n_filters_conv2, mid_kernel_size, 1))
        self.add_module('act2', act())
        self.add_module('pool2', pool(2))
        self.add_module('conv3', conv(n_filters_conv2, n_filters_conv3, 5, 1))
        self.add_module('flatten', flatten)
        self.add_module('act3', act())
        self.add_module('lin1', torch.nn.Linear(n_filters_conv3 * 1 * 1, 84))  
        self.add_module('act4', act())
        self.add_module('linout', torch.nn.Linear(84, n_out))
        
        
class PrunedCancerNet_fc(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(PrunedCancerNet_fc, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x