# this script is just to recover the data from the experiment if it was interrupted/ crashed and continue the sparsification process
from models import ResNet   
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader



from torchsummary import summary
from io import StringIO
from torchviz import make_dot

from models import ResNet
import torch
import torch.nn as nn
from torchsummary import summary
from torchviz import make_dot
from sparsify import sparse_strategy, magnitude_pruning, random_strategy
import copy
import pickle as pkl
import wandb
# Create the model
model = ResNet(18, 64)

# Load the model weights
model.load_state_dict(torch.load("results/ResNet_64_DiagLaplace_scalar_100_wp/model.pt"))

num_workers = 0
# dataloader
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

tforms = [transforms.ToTensor(),
        transforms.Normalize(mean, std)]
tforms_test = transforms.Compose(tforms)
tforms_train = tforms_test
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=tforms_train)
valid_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=tforms_test)


train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=num_workers)
valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers=num_workers)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

la = pkl.load(open("results/ResNet_64_DiagLaplace_scalar_100_wp/laplace.pkl", "rb"))
sparsities = [20,40,60,70,75,80,85,90,95,99]
sparse_list = {'laplacekron':{'model': copy.deepcopy(model), 'function': sparse_strategy ,'sparsities': sparsities, 'la': la},
                'magnitude':{'model': copy.deepcopy(model), 'function': magnitude_pruning ,'sparsities': sparsities},
                'random':{'model': copy.deepcopy(model), 'function': random_strategy ,'sparsities': sparsities}
                }


for sparse_name, sparse_dict in sparse_list.items():
    with wandb.init(reinit=True, id=wandb.util.generate_id(), project="BNN_Sparse", entity="xxxxxx", name=f"ResNet_64_DiagLaplace_scalar_100_wp_{sparse_name}"):
        sparse_dict['function'](sparse_dict['la'],sparse_dict['model'], valid_loader, sparse_dict['sparsities'])
        
    wandb.finish() 