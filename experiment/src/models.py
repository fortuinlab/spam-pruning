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

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
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

class LeNet(nn.Sequential):

    def __init__(self, in_channels=1, n_out=10, activation='relu', n_pixels=28,
                 augmented=False):
        super().__init__()
        mid_kernel_size = 3 if n_pixels == 28 else 5
        act = get_activation(activation)
        conv = nn.Conv2d
        pool = nn.MaxPool2d
        flatten = nn.Flatten(start_dim=2) if augmented else nn.Flatten(start_dim=1)
        self.add_module('conv1', conv(in_channels, 6, 5, 1))
        self.add_module('act1', act())
        self.add_module('pool1', pool(2))
        self.add_module('conv2', conv(6, 16, mid_kernel_size, 1))
        self.add_module('act2', act())
        self.add_module('pool2', pool(2))
        self.add_module('conv3', conv(16, 120, 5, 1))
        self.add_module('flatten', flatten)
        self.add_module('act3', act())
        self.add_module('lin1', torch.nn.Linear(120*1*1, 84))
        self.add_module('act4', act())
        self.add_module('linout', torch.nn.Linear(84, n_out))

class LanguageTransformer(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)
        self.ntoken = ntoken

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        output = output.view(-1, self.ntoken)
        return output
    


class VisionTransformer(nn.Module):
    def __init__(self, image_size=32, patch_size=8, num_classes=10, dim=64, depth=6, heads=8, mlp_dim=128, channels=1):
        super(VisionTransformer, self).__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.patch_dim = channels * patch_size ** 2
        self.num_patches = (image_size // patch_size) ** 2  # Ensure this calculates to 16 for 32x32 images with 8x8 patches

        self.patch_to_embedding = nn.Linear(self.patch_dim, dim)
        self.position_embeddings = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim) for _ in range(depth)
        ])

        self.to_cls_token = nn.Identity()
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x):
        # Convert image patches to embeddings
        x = self.patch_to_embedding(x.view(x.size(0), self.num_patches, self.patch_dim)).transpose(1, 2)
        x = x.transpose(1, 2)  # Changing shape from [128, 64, 16] to [128, 16, 64]
        x = torch.cat((self.cls_token.repeat(x.size(0), 1, 1), x), dim=1)  # Concatenate along dimension 1
        x += self.position_embeddings

        # Transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        # Classifier
        x = self.to_cls_token(x[:, 0])
        return self.classifier(x)
    

class MLPMixerLayer(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(MLPMixerLayer, self).__init__()
        
        self.channel_mixing = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
        
    def forward(self, x):
        x = x + self.channel_mixing(x)
        return x

class MLPMixer(nn.Module):
    def __init__(self, dim=784, num_classes=10, hidden_dim=256, num_blocks=2):
        super(MLPMixer, self).__init__()
        
        self.embedding = nn.Linear(dim, dim)
        
        # Stack of Mixer layers
        self.mixer_layers = nn.ModuleList([])
        for _ in range(num_blocks):
            self.mixer_layers.append(MLPMixerLayer(dim, hidden_dim))
        
        self.head = nn.Linear(dim, num_classes)
        
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.mixer_layers:
            x = layer(x)
        return self.head(x)