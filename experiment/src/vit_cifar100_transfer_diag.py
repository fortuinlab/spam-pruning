
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

from transformers import ViTFeatureExtractor, ViTForImageClassification




class CustomViTForImageClassification(nn.Module):
    def __init__(self, model_name, num_classes):
        super(CustomViTForImageClassification, self).__init__()
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name, do_rescale=False)
        self.model = ViTForImageClassification.from_pretrained(model_name)
        self.model.classifier = nn.Linear(in_features=self.model.config.hidden_size, out_features=num_classes, bias=True)

    def forward(self, x):
        # Apply the feature extractor
        x = x.clamp(0, 1)
        x = x.permute(0, 2, 3, 1)
        x = self.feature_extractor(images=x, return_tensors="pt")["pixel_values"]
        x = x.to(next(self.model.parameters()).device)
        # Get the logits
        outputs = self.model(x)
        return outputs.logits



def config_Wb(config):
    unique_id = wandb.util.generate_id()
    run_name = f"ViT_huggingg_Cifar100_{config['num_epochs']}"
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


                          
                
if __name__ == "__main__":
    seed = np.random.randint(0, 1000)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    
    config = {
        "model_name": "ViT_huggingface",
        "dataset_name": "Cifar100",
        "optimizer": "adamw",
        "batch_size": 128,
        "num_epochs": 20,
        "lr": 0.0001,
        "weight_decay": 0.0,
        "seed": seed,
        "laplace": {
            "laplace_type": "DiagLaplace",
            "prior_structure": "diagonal",
            "marglik_frequency": 5,
            "num_epochs_burnin": 10,
            "lr_min": 1e-6,
            "n_hypersteps": 50,
        }
    }
    run_name_orig = config_Wb(config)
    # cifar100
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((224, 224), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2)
    
   
    # load vit hugging face 

    
    model = CustomViTForImageClassification('google/vit-base-patch16-224-in21k', 100)

    
    for param in model.parameters():
        param.requires_grad = True
        
    
    
    
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
                                        n_hypersteps= config['laplace']['n_hypersteps'],
                                        lr_min = config['laplace']['lr_min'],
                                        optimizer= config['optimizer'],
                                        laplace=lap,
                                        temperature = 1,
                                        n_epochs= config['num_epochs'],
                                        n_epochs_burnin=config['laplace']['num_epochs_burnin'],
                                        prior_structure= config['laplace']['prior_structure'], 
                                        marglik_frequency= config['laplace']['marglik_frequency'],                            
                                        log_wandb = True,
                                    )
        print(val_perf)
        
        
        
        num_classes_brier = 100
        tunemethod = "map"
        config['tune'] = False # one shot
        sparsities = [20,40,60,70,75,80,85,90,95,99]
        args_sparse= {
            'num_classes': num_classes_brier,
            'prior_structure': config['laplace']['prior_structure'],
            'tune_epochs_burnin': 11 if tunemethod == "map" else 0,
            'marglik_frequency': config['laplace']['marglik_frequency'],
            'fine_tune': config["tune"],
            'tune_epochs': 2,
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
            run_name = f"{run_name}_sub_{sparse_name}_finetune_{args_sparse['tune_epochs']}_{tunemethod}" if args_sparse['fine_tune'] == True else f"{run_name}_New_sub_{sparse_name}"
            

            
            with wandb.init(reinit=True, id=wandb.util.generate_id(),config = cfgs, project='BNN_Sparse', entity="xxxxxx", name=run_name):
                wandb.config.update(args_sparse, allow_val_change=True)
                wandb.config.update({'sparsification_method': sparse_name}, allow_val_change=True)
                models_stats = sparse_dict['function'](la=sparse_dict['la'],model = sparse_dict['model'], test_loader= test_loader,
                                        train_loader = train_loader, sparsities = sparse_dict['sparsities'],args= args_sparse)

                

        
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
        