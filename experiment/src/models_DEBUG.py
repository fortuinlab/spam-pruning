import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import nn, Tensor
import torchvision.models as models
from asdfghjkl.operations import Bias, Scale

#from asdfghjkl.operations.conv_aug import Conv2dAug

class CancerNet_fc(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CancerNet_fc, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class BostonReg_fc(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BostonReg_fc, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
    
class ResNet18(nn.Module):
    def __init__(self, num_classes):
        # number of classes for cifar 10 is 10
        super(ResNet18, self).__init__()
        self.resnet18 = models.resnet18(pretrained=False)
        num_filters = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_filters, num_classes)

    def forward(self, x):
        x = self.resnet18(x)
        return x

class MNIST_FC(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MNIST_FC, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class Cifar10_fc(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Cifar10_fc, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
class Cifar10_CNN(nn.Module):
    def __init__(self, num_classes):
        super(Cifar10_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc = nn.Linear(32*8*8, num_classes)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1) # flatten the tensor
        out = self.fc(out)
        return out
    
def conv3x3(in_planes, out_planes, stride=1, augmented=False):
    """3x3 convolution with padding"""
    #Conv2d = Conv2dAug if augmented else nn.Conv2d
    Conv2d = nn.Conv2d
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                  padding=1, bias=False)


class FixupBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, augmented=False):
        super(FixupBasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.augmented = augmented
        self.bias1a = Bias()
        self.conv1 = conv3x3(inplanes, planes, stride, augmented=augmented)
        self.bias1b = Bias()
        self.relu = nn.ReLU(inplace=True)
        self.bias2a = Bias()
        self.conv2 = conv3x3(planes, planes, augmented=augmented)
        self.scale = Scale()
        self.bias2b = Bias()
        self.downsample = downsample

    def forward(self, x):
        identity = x

        biased_x = self.bias1a(x)
        out = self.conv1(biased_x)
        out = self.relu(self.bias1b(out))

        out = self.conv2(self.bias2a(out))
        out = self.bias2b(self.scale(out))

        if self.downsample is not None:
            identity = self.downsample(biased_x)
            cat_dim = 2 if self.augmented else 1
            identity = torch.cat((identity, torch.zeros_like(identity)), cat_dim)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    FixupResnet-depth where depth is a `4 * 2 * n + 2` with `n` blocks per residual layer.
    The two added layers are the input convolution and fully connected output.
    """

    def __init__(self, depth, num_classes=10, in_planes=16, in_channels=3, augmented=False,
                 last_logit_constant=False):
        super(ResNet, self).__init__()
        n_out = num_classes if not last_logit_constant else num_classes - 1
        self.llc = last_logit_constant
        assert (depth - 2) % 8 == 0, 'Invalid ResNet depth, has to conform to 6 * n + 2'
        layer_size = (depth - 2) // 8
        layers = 4 * [layer_size]
        self.num_layers = 4 * layer_size
        self.inplanes = in_planes
        self.augmented = augmented
        #AdaptiveAvgPool2d = AdaptiveAvgPool2dAug if augmented else nn.AdaptiveAvgPool2d
        AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d
        self.conv1 = conv3x3(in_channels, in_planes, augmented=augmented)
        self.bias1 = Bias()
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(FixupBasicBlock, in_planes, layers[0])
        self.layer2 = self._make_layer(FixupBasicBlock, in_planes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(FixupBasicBlock, in_planes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(FixupBasicBlock, in_planes * 8, layers[3], stride=2)
        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(start_dim=2 if augmented else 1)
        self.bias2 = Bias()
        self.fc = nn.Linear(in_planes * 8, n_out)
        #if last_logit_constant: # dont have the implemtation but it is not even needed
        #    self.constant_logit = ConstantLastLogit(augmented=augmented)

        for m in self.modules():
            if isinstance(m, FixupBasicBlock):
                nn.init.normal_(m.conv1.weight,
                                mean=0,
                                std=np.sqrt(2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** (-0.5))
                nn.init.constant_(m.conv2.weight, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        #AvgPool2d = AvgPool2dAug if self.augmented else nn.AvgPool2d
        AvgPool2d = nn.AvgPool2d
        if stride != 1:
            downsample = AvgPool2d(1, stride=stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, augmented=self.augmented))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(planes, planes, augmented=self.augmented))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(self.bias1(x))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(self.bias2(x))
        if self.llc:
            x = self.constant_logit(x)

        return x
    

class WRNFixupBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, augmented=False, fixup=True):
        super(WRNFixupBasicBlock, self).__init__()
        self.bias1 = Bias() if fixup else nn.Identity()
        self.relu1 = nn.ReLU(inplace=True)
        basemodul = nn.Conv2d#Conv2dAug if augmented else nn.Conv2d
        self.augmented = augmented
        self.conv1 = basemodul(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bias2 = Bias() if fixup else nn.Identity()
        self.relu2 = nn.ReLU(inplace=True)
        self.bias3 = Bias() if fixup else nn.Identity()
        self.conv2 = basemodul(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bias4 = Bias() if fixup else nn.Identity()
        self.scale1 = Scale() if fixup else nn.Identity()
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and basemodul(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bias1(x))
        else:
            out = self.relu1(self.bias1(x))
        if self.equalInOut:
            out = self.bias3(self.relu2(self.bias2(self.conv1(out))))
        else:
            out = self.bias3(self.relu2(self.bias2(self.conv1(x))))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.bias4(self.scale1(self.conv2(out)))
        if not self.equalInOut:
            return torch.add(self.convShortcut(x), out)
        else:
            return torch.add(x, out)


class WRNFixupNetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, augmented=False, fixup=True):
        super(WRNFixupNetworkBlock, self).__init__()
        self.augmented = augmented
        self.fixup = fixup
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, self.augmented, self.fixup))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)



class WideResNet(nn.Module):
    def __init__(self, depth=16, widen_factor=4, num_classes=10, dropRate=0.0, augmented=False, fixup=True):
        super(WideResNet, self).__init__()
        n_out = num_classes
        self.fixup = fixup
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = WRNFixupBasicBlock
        # 1st conv before any network block
        self.num_layers = n * 3
        basemodul = nn.Conv2d #Conv2dAug if augmented else nn.Conv2d
        self.augmented = augmented
        self.conv1 = basemodul(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bias1 = Bias() if fixup else nn.Identity()
        # 1st block
        self.block1 = WRNFixupNetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, augmented=augmented, fixup=fixup)
        # 2nd block
        self.block2 = WRNFixupNetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, augmented=augmented, fixup=fixup)
        # 3rd block
        self.block3 = WRNFixupNetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, augmented=augmented, fixup=fixup)
        # global average pooling and classifier
        self.bias2 = Bias() if fixup else nn.Identity()
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(8)#AvgPool2dAug(8) if augmented else nn.AvgPool2d(8)
        self.fc = nn.Linear(nChannels[3], n_out)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, WRNFixupBasicBlock):
                conv = m.conv1
                k = conv.weight.shape[0] * np.prod(conv.weight.shape[2:])
                nn.init.normal_(conv.weight,
                                mean=0,
                                std=np.sqrt(2. / k) * self.num_layers ** (-0.5))
                nn.init.constant_(m.conv2.weight, 0)
                if m.convShortcut is not None:
                    cs = m.convShortcut
                    k = cs.weight.shape[0] * np.prod(cs.weight.shape[2:])
                    nn.init.normal_(cs.weight,
                                    mean=0,
                                    std=np.sqrt(2. / k))
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.bias1(self.conv1(x))
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(out)
        out = self.pool(out)
        if self.augmented:
            out = out.flatten(start_dim=2)
        else:
            out = out.flatten(start_dim=1)
        out = self.fc(self.bias2(out))
        return out



class PositionalEncoding(nn.Module):

    def __init__(self, d_model: 200, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class LanguageTransformer(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)
        self.ntoken = ntoken
        self.bptt = 35
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[batch_size * seq_len]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[batch_size * seq_len, ntoken]``
        """
        batch_size = src.size(0) // self.bptt
        if batch_size!=0:
            seq_len = src.size(0) // batch_size 
        else:
            seq_len = src.size(0)
        output_list = []

        for i in range(batch_size):
            input_batch = src[i * seq_len: (i + 1) * seq_len]
            input_batch = input_batch.unsqueeze(0)  # Add a batch dimension

            input_batch = self.embedding(input_batch) * math.sqrt(self.d_model)
            input_batch = input_batch.permute(1, 0, 2)  # Swap dimensions for transformer encoder
            input_batch = self.pos_encoder(input_batch)

            output_batch = self.transformer_encoder(input_batch, src_mask)
            output_batch = self.linear(output_batch.view(-1, self.d_model))
            output_list.append(output_batch)

        output = torch.cat(output_list, dim=0)
        return output