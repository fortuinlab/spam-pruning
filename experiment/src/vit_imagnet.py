
import argparse
import os
import shutil
import time
import wandb
import random
import numpy as np

import copy

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from sparsify_v2 import (sparse_strategy, magnitude_pruning, random_strategy, 
                         SNIP_strategy, GraSP_strategy)

from marglikopt import marglik_optimization
from laplace import KronLaplace, DiagLaplace
from asdfghjkl.operations import Bias, Scale
from einops import rearrange
from einops.layers.torch import Rearrange



def config_Wb(config):
    unique_id = wandb.util.generate_id()
    run_name = f"ViT_Imagenet_{config['num_epochs']}"
    if config['laplace']['num_epochs_burnin'] > config['num_epochs']:
        run_name += '_MAP'
    else:
        run_name += f"_{config['laplace']['laplace_type']}_{config['laplace']['prior_structure']}"
        
    
    use_map = False
    seed = config['seed']
    # excluding the laplace part from the config as we are using the new maglikopt script

    to_log = {
        

        'model_name': config['model_name'],
        'dataset_name': config['dataset_name'],
        'optimizer': config['optimizer'],

        'batch_size': config['batch_size'],
        'num_epochs': config['num_epochs'],
        'lr': config['lr'],
        'weight_decay': config['weight_decay'],
        'seed': seed,
        "marglik_param": {
                "laplace": "MAP" if use_map else config['laplace']['laplace_type'],
                "prior_structure": "MAP" if use_map else config['laplace']['prior_structure'],
            }
        
    }
       
    wandb.init(id = unique_id, name=run_name, project='BNN_Sparse', entity="xxxxxx", config=to_log)
    return run_name





def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(patches, temperature=10000, dtype=torch.float32, augmented=False):
    if augmented:
        _, _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype
    else:
        _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device=device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


def ViTFeedForward(dim, hidden_dim, fixup=False):
        return nn.Sequential(
            Bias() if fixup else nn.Identity(),
            nn.Linear(dim, hidden_dim,bias=False),
            Bias() if fixup else nn.Identity(),
            nn.GELU(),
            Bias() if fixup else nn.Identity(),
            nn.Linear(hidden_dim, dim,),
            Scale() if fixup else nn.Identity()
        )


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, fixup=False, augmented=False):
        super().__init__()
        self.shift = Bias() if fixup else nn.Identity()
        self.augmented = augmented
        inner_dim = dim_head * heads
        self.heads = heads
        self._scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim*3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.scale = Scale() if fixup else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(self.shift(x)).chunk(3, dim=-1)
        bspec = 'b m' if self.augmented else 'b'
        q, k, v = map(lambda t: rearrange(t, f'{bspec} n (h d) -> {bspec} h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self._scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, f'{bspec} h n d -> {bspec} n (h d)')
        return self.scale(self.to_out(out))


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, fixup, augmented):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, fixup=fixup, augmented=augmented),
                ViTFeedForward(dim, mlp_dim, fixup=fixup)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    """Simple vision transformer SimpleViT."""
    def __init__(self, image_size=32, patch_size=4, num_classes=10, dim=512, depth=6, heads=8,
                 mlp_dim=512, channels=3, dim_head=64, fixup=False, augmented=False):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        patch_dim = channels * patch_height * patch_width
        n_out = num_classes
        self.augmented = augmented
        bspec = 'b m' if augmented else 'b'
        self.bspec = bspec
        self.to_patch_embedding = nn.Sequential(
            Rearrange(f'{bspec} c (h p1) (w p2) -> {bspec} h w (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim, bias=False),
        )
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, fixup, augmented)
        self.to_latent = Bias() if fixup else nn.Identity()
        self.linear_head = nn.Linear(dim, n_out, bias=False)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        pe = posemb_sincos_2d(x, augmented=self.augmented)
        x = rearrange(x, f'{self.bspec} ... d -> {self.bspec} (...) d') + pe
        x = self.transformer(x)
        x = x.mean(dim=2 if self.augmented else 1)
        x = self.to_latent(x)
        x = self.linear_head(x)
        return x

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, Bias, Scale)):
                module.reset_parameters()
                
def subsample_loader(data_loader, batches=5):
    """
    Yields a limited number of batches from the DataLoader.
    :param data_loader: The DataLoader instance to subsample from.
    :param batches: The number of batches to yield.
    """
    for i, data in enumerate(data_loader):
        if i >= batches:
            break
        yield data               

class MixUpDataOnly(object):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, batch):
        data, targets = zip(*batch)
        data = torch.stack(data)
        targets = torch.LongTensor(targets)

        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = data.size(0)
        index = torch.randperm(batch_size)

        mixed_data = lam * data + (1 - lam) * data[index, :]
        # Return original targets instead of mixed targets
        return mixed_data, targets
             
if __name__ == "__main__":
    seed = np.random.randint(0, 1000)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    
    config = {
        "model_name": "ViT",
        "dataset_name": "Imagenet",
        "optimizer": "adamw",
        "batch_size": 256, 
        "num_epochs": 100,
        "lr": 0.001,
        "weight_decay": 0.0,
        "seed": seed,
        "laplace": {
            "laplace_type": "DiagLaplace",
            "prior_structure": "scalar",
            "marglik_frequency": 5,
            "num_epochs_burnin": 101,
            "lr_min": 1e-06,
            "n_hypersteps": 50,
        }
    }
    
    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    run_name_orig = config_Wb(config)

    transform_test= transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    
    train_dataset = datasets.ImageFolder(
    root='/nfs/shared/imagenet2012/train', 
    transform=transform
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0,collate_fn=MixUpDataOnly(alpha=1.0  ))
    
    test_dataset = datasets.ImageFolder(
    root='/nfs/shared/imagenet2012/val',
    transform=transform_test
    )
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    # length of train loader
    #print(len(train_loader))
    
    for i, (images, labels) in enumerate(train_loader):
        print(f"Batch {i+1} labels: {labels.tolist()}")
        if i == 0:  # Change 0 to the number of batches you want to print minus one
            break
    
    #model = ViT(image_size=224, patch_size=16, num_classes=1000, dim=768, depth=12, heads=16, mlp_dim=3072,
    #            channels=3)
    # smaller vit
    model = ViT(image_size=224, patch_size=16, num_classes=1000, dim=256, depth=6, heads=8, mlp_dim=512,
                channels=3)
    
    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #if torch.cuda.device_count() > 1:
    #    model = torch.nn.DataParallel(model)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    
    train = 'marglik'
    
    # print model number of parameters
    #print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    # weights only without bias
    #print(f"Number of weights: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    
    lap = KronLaplace if config['laplace']['laplace_type'] == "KronLaplace" else DiagLaplace
    
    if train == 'marglik':
        print('Training with Marglik')
        la, model, margliks, val_perf = marglik_optimization(           
                                        model=model, train_loader=train_loader,
                                        valid_loader= test_loader,likelihood="classification",
                                        lr= config['lr'],
                                        #n_hypersteps= config['laplace']['n_hypersteps'],
                                        #lr_min = config['laplace']['lr_min'],
                                        optimizer= config['optimizer'],
                                        laplace=lap,
                                        temperature = 1,
                                        n_epochs= config['num_epochs'],
                                        n_epochs_burnin=config['laplace']['num_epochs_burnin'],
                                        prior_structure= config['laplace']['prior_structure'],                             
                                        log_wandb = True,
                                    )
        print(val_perf)
        
        
        
        num_classes_brier = 1000
        tunemethod = "map"
        config['tune'] = False # one shot
        sparsities = [20,40,60,70,75,80,85,90,95,99]
        args_sparse= {
            'num_classes': num_classes_brier,
            'prior_structure': config['laplace']['prior_structure'],
            'tune_epochs_burnin': 11 if tunemethod == "map" else 0,
            'marglik_frequency': config['laplace']['marglik_frequency'],
            'fine_tune': config["tune"],
            'tune_epochs': 10,
            'lr': config['lr']*0.1
        }
  
        sparse_list = {'laplacekron':{'model': copy.deepcopy(model), 'function': sparse_strategy ,'sparsities': sparsities, 'la': la},
                        'magnitude':{'model': copy.deepcopy(model), 'function': magnitude_pruning ,'sparsities': sparsities, 'la': la},
                        'random':{'model': copy.deepcopy(model), 'function': random_strategy ,'sparsities': sparsities, 'la': la},
                        'SNIP':{'model': copy.deepcopy(model), 'function': SNIP_strategy ,'sparsities': sparsities, 'la': la},
                        'GraSP':{'model': copy.deepcopy(model), 'function': GraSP_strategy ,'sparsities': sparsities, 'la': la},
                        }
        
            
        for sparse_name, sparse_dict in sparse_list.items():
            cfgs = config
            run_name = run_name_orig
            run_name = f"{run_name}_sub_{sparse_name}_finetune_{args_sparse['tune_epochs']}_{tunemethod}" if args_sparse['fine_tune'] == True else f"{run_name}_sub_{sparse_name}"
            

            try:
                with wandb.init(reinit=True, id=wandb.util.generate_id(),config = cfgs, project='BNN_Sparse', entity="xxxxxx", name=run_name):
                    wandb.config.update(args_sparse, allow_val_change=True)
                    wandb.config.update({'sparsification_method': sparse_name}, allow_val_change=True)
                    models_stats = sparse_dict['function'](la=sparse_dict['la'],model = sparse_dict['model'], test_loader= test_loader,
                                            train_loader = train_loader, sparsities = sparse_dict['sparsities'],args= args_sparse)
            except:
                print(f"Error in {sparse_name}")
                continue
                

        
        #posterior_precision = la.posterior_precision.diag()
    
    
    """else:
        for epoch in range(10):
            print(f'Epoch {epoch+1}/{10}')
            model.train()
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                if (i+1) % 100 == 0:
                    print(f'Epoch [{epoch+1}/{10}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        print('Finished Training')
        
        # Test the model on test data accuracy
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')
        print('Finished Testing')"""
        