# Built-in 
import os
import time
import logging
import pickle
import copy

# libraries
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, dataset
from torchvision import transforms, datasets
import torchvision
import torchvision.transforms as transforms
from laplace import KronLaplace, DiagLaplace
import numpy as np
import pandas as pd
import wandb
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# own  modules
from models import (CancerNet_fc, BostonReg_fc, ResNet18, MNIST_FC, Cifar10_fc, 
                    Cifar10_CNN, ResNet, WideResNet, LanguageTransformer, LeNet, VisionTransformer, MLPMixer)

from wide_resfixup import FixupWideResNet

from datasets_custom import (CancerDataset, CancerDataset_supported, BostonHousingDataset, 
                      CIFAR10Dataset, MNISTDataset, RotatedCIFAR100, RotatedMNIST)
from utils import check_sparsity

from marglikopt_sp_new import marglik_optimization

import logging
logging.basicConfig(level=logging.INFO)

# seeds 

seed = np.random.randint(0, 200)
torch.manual_seed(seed)
np.random.seed(seed)



margopt = True

class train_cfg():
    def __init__(self, config, cfg_name, sparsewith):
        self.aug = True
        self.config = config
        self.cfg_name = cfg_name
        self.run_name = None  
        self.pickle_laplace = False
        self.sparse_method = config.get('sparse_method', None)
        self.sparse_exact_same = True
        self.use_map = self.config.get('use_map', False)
        self.infere_exp = True
        self.resnet_inplanes = self.config.get('resnet_inplanes', 32)
        # online sparsity linked parameters
        self.update_interval = 10
        self.target_sparsity = 0.99
        self.sp_method = sparsewith
        # for previous generated cfgs

        # check if ['laplace']['n_epochs_burnin'] is not existent then set it to 0
        # for old cfgs adding these checks  to avoid errors
        if 'n_epochs_burnin' not in self.config['laplace']:
            self.config['laplace']['n_epochs_burnin'] = 0
        if 'train_marglik' not in self.config:
            self.config['train_marglik'] = True
        if 'lr_min' not in self.config['laplace']:
            self.config['laplace']['lr_min'] = None
        if 'marglik_frequency' not in self.config['laplace']:
            self.config['laplace']['marglik_frequency'] = 1
        if 'n_hypersteps' not in self.config['laplace']:
            self.config['laplace']['n_hypersteps'] = 100            
        if 'lr_hyp' not in self.config['laplace']:
            self.config['laplace']['lr_hyp'] = 1e-1    
        if 'prior_prec_init' not in self.config['laplace']:
            self.config['laplace']['prior_prec_init'] = 1.
            
        if 'temperature' not in self.config['laplace']:
            self.temperature = 1
        else:
            self.config['laplace']['temperature'] = 1 / self.config['laplace']['temperature']
            self.temperature = self.config['laplace']['temperature']


    def get_run_name(self, epoch=None):
        """Return the name of the current run based on the model and epoch number."""
        model_name = self.config['model_name']

        if self.config['train_marglik']:
            #backend = self.config['laplace']['backend']
            #hessian = self.config['laplace']['hessian_structure']
            prior = self.config['laplace']['prior_structure']
            epoch = self.config['num_epochs']
            laplace_type = self.config['laplace']['laplace_type']
            sparse_method = self.sparse_method
            online_score = "LaplaceOnline" if self.sp_method == "post_and_weights" else "PostOnline"
            if laplace_type is not None:
                if sparse_method is not None:
                    return f'{model_name}_{laplace_type}_{prior}_{epoch}_Sparse_{sparse_method}'
                else:
                    if self.use_map or self.config['laplace']['n_epochs_burnin']> epoch :
                        self.use_map = True
                        return f'{model_name}_{self.config["dataset_name"]}_{epoch}_MAP'
                    else:
                        if self.resnet_inplanes == 64 and self.config['model_name'] == 'ResNet':
                            return f'{model_name}_64_{laplace_type}_{prior}_{epoch}_wp_{online_score}'
                        else:
                            
                            return f'{model_name}_{self.config["dataset_name"]}_{laplace_type}_{prior}_{epoch}_{online_score}'
                        
            else:
                if sparse_method is not None:
                    return f'{model_name}_{prior}_{epoch}_{sparse_method}'
                else:
                    return f'{model_name}_{prior}_{epoch}'
        else:
            if epoch is not None:
                return f'{model_name}_{epoch}'
            else:
                return model_name
            
    def config_Wb(self):
        unique_id = wandb.util.generate_id()
        self.run_name = self.get_run_name()
        # excluding the laplace part from the config as we are using the new maglikopt script
        traintype = "Map "if self.use_map else "Marglik"
        to_log = {
            'cfg_name': self.cfg_name,

            'model_name': self.config['model_name'],
            'dataset_name': self.config['dataset_name'],
            'optimizer': self.config['optimizer'],

            'batch_size': self.config['batch_size'],
            'num_epochs': self.config['num_epochs'],
            'lr': self.config['lr'],
            'weight_decay': self.config['weight_decay'],
            'seed': seed,
            "Training": traintype,
            "marglik_param": {
                    "laplace": "MAP" if self.use_map else self.config['laplace']['laplace_type'],
                    "prior_structure": "MAP" if self.use_map else self.config['laplace']['prior_structure'],
                }
            
        }
       
        wandb.init(id = unique_id, name=self.run_name, project=self.config['wandb']['project'], entity=self.config['wandb']['entity'], config=to_log)


    def Dataloader(self):

        
        dataset_name = self.config['dataset_name']
        batch_size = self.config['batch_size']
        num_workers = 0 if self.config['device'] == 'cuda' else 4

        if dataset_name == "breast_cancer":
            self.likelihood = "classification"
            train_dataset = CancerDataset(train=True)
            valid_dataset = CancerDataset(train=False)

        elif dataset_name == "boston_housing":
            self.likelihood = "regression"
            data = BostonHousingDataset(data=pd.read_csv("./data/housing.csv"))
            train_dataset = torch.utils.data.Subset(data, list(range(0, len(data) - 100)))
            valid_dataset = torch.utils.data.Subset(data, list(range(len(data) - 100, len(data))))

        elif dataset_name == "Cifar10":
            self.likelihood = "classification"
            if self.config['model_name'] == 'ResNet' or self.config['model_name'] == 'LeNet':
                mean = [x / 255 for x in [125.3, 123.0, 113.9]]
                std = [x / 255 for x in [63.0, 62.1, 66.7]]
                tforms = [transforms.ToTensor(),
                        transforms.Normalize(mean, std)]
                tforms_test = transforms.Compose(tforms)
                if self.aug:
                    tforms_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop(32, padding=4)]
                                        + tforms)
                else:
                    tforms_train = tforms_test
                
                train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=tforms_train)
                valid_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=tforms_test)
            elif self.config["model_name"]== "MLPMixer":
                transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: x.view(-1))
                    ])

                train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
                valid_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
            else:
                
                train_dataset = CIFAR10Dataset(train=True)
                valid_dataset = CIFAR10Dataset(train=False)
        elif dataset_name == "Cifar10":
            self.likelihood = "classification"
            if self.config['model_name'] == 'ResNet' or self.config['model_name'] == 'LeNet' or self.config['model_name'] == 'WideResNet':
                    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
                    std = [x / 255 for x in [63.0, 62.1, 66.7]]

                    tforms = [transforms.ToTensor(),
                            transforms.Normalize(mean, std)]
                    tforms_test = transforms.Compose(tforms)
                    
                    tforms_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop(32, padding=4)]
                                        + tforms)
                    print("augmented data")
                    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=tforms_train)
                    valid_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=tforms_test)
        elif dataset_name == "RotatedCifar100":
            self.likelihood = "classification"
            train_dataset = RotatedCIFAR100(root='./data',degree=0, train=True, download=True, transform=transforms.ToTensor())
            valid_dataset = RotatedCIFAR100(root='./data',degree=0, train=False, download=True, transform=transforms.ToTensor())

        elif dataset_name == "mnist" and self.config['model_name']!= "LeNet":
            self.likelihood = "classification"
            train_dataset = MNISTDataset(root='data', train=True, transform=transforms.ToTensor())
            valid_dataset = MNISTDataset(root='data', train=False, transform=transforms.ToTensor())
        
        elif dataset_name == "mnist" and self.config['model_name']== "LeNet":
            self.likelihood = "classification"
            train_dataset = RotatedMNIST(root='./data', degree=0, train=True, download=True, transform=transforms.ToTensor())
            valid_dataset = RotatedMNIST(root='data', degree=0, train=False, download=True, transform=transforms.ToTensor())

        elif dataset_name == "FashionMnist":
            self.likelihood = "classification"
            if self.config['model_name'] == 'VisionTransformer':
                transform = transforms.Compose([
                transforms.Resize((32, 32)),  # Resize the images to 32x32
                transforms.ToTensor(),
                ])
                train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
                valid_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
            else:
                train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
                valid_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
        else:
            raise ValueError("Invalid dataset name!")
        """elif dataset_name == "WikiText2":
            self.likelihood = "classification"
            
            data = DataProcessorWikiText2()
            data.setup_data()
            self.train_loader = data.train_data
            self.valid_loader = data.val_data
            self.vocab= data.vocab
            self.vocab_size = len(self.vocab)"""
    
        if dataset_name != "WikiText2":
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            self.valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    def load_encoder(self):

        model_name = self.config['model_name']
        dataset_name = self.config['dataset_name']
        likelihood = self.config.get('likelihood', 'regression')

        if model_name == "CancerNet_fc":
            self.model = CancerNet_fc(input_size=30, hidden_size=100, output_size=2)

        elif model_name == "BostonReg_fc":
            self.model = BostonReg_fc(input_size=13, hidden_size=100, output_size=1)

        elif model_name == "MNIST_FC":
            self.model = MNIST_FC(784,256,10)

        elif model_name == "ResNet18":
            if dataset_name == "cifar10":
                num_classes = 10
            else:
                num_classes = 2
            self.model = ResNet18(num_classes=num_classes)
            
        elif model_name == "ResNet":
            self.model = ResNet(depth=18, in_planes= self.resnet_inplanes,num_classes=10)

        elif model_name == "WideResNet":
            if dataset_name == "Cifar10":
                self.model =  FixupWideResNet(16, 4, 10, dropRate=0)
            elif dataset_name == "Cifar100":
                self.model = FixupWideResNet(16, 4, 100, dropRate=0)


        elif model_name == "ResNet106":
            if dataset_name == "Cifar100":
                self.model = ResNet(depth=106, in_planes= self.resnet_inplanes,num_classes=100)

        elif model_name == "ResNet50":
            if dataset_name == "Cifar100":
                self.model = ResNet(depth=50, in_planes= self.resnet_inplanes,num_classes=100)
            elif dataset_name == "RotatedCifar100":
                self.model = ResNet(depth=50, in_planes= self.resnet_inplanes,num_classes=100)

        elif model_name == "Cifar10_fc":
            self.model = Cifar10_fc(input_size=3072, hidden_size=100, num_classes=10)
        elif model_name == "Cifar10_CNN":
            self.model = Cifar10_CNN(num_classes=10)
        
        elif model_name == "LeNet":
            if dataset_name == "mnist":
                self.model = LeNet(n_out=10)
            elif dataset_name == "FashionMnist":
                self.model = LeNet(n_out=10)
            elif dataset_name == "Cifar10":
                self.model = LeNet(in_channels=3, n_out=10, activation='relu', n_pixels=32)
            
        elif model_name == "MLPMixer":
            if dataset_name == "mnist":
                self.model = MLPMixer(dim=784, num_classes=10, hidden_dim=256, num_blocks=2)
            elif dataset_name == "Cifar10": 
                self.model = MLPMixer(dim=3072, num_classes=10, hidden_dim=256, num_blocks=2)

        elif model_name == "LanguageTransformer":
            self.model = LanguageTransformer(ntoken=33278, d_model=200, nhead=2, d_hid=200, nlayers=2, dropout=0.2)
        
        elif model_name == "VisionTransformer":
            if self.config['dataset_name'] == "FashionMnist":
                self.model = VisionTransformer(channels=1)
            
        else:
            raise ValueError("Invalid model name!")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def set_optimizer(self):
        optimizer_name = self.config['optimizer']
        lr = self.config['lr']
        weight_decay = self.config['weight_decay']
        model_name = self.config['model_name']

        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        else:
            raise ValueError("Invalid optimizer name!")

        self.criterion =  torch.nn.MSELoss() if self.likelihood == "regression" else torch.nn.CrossEntropyLoss()

        self.optimizer = optimizer
        
    def train(self):
        
        
        self.save_path_inference = Path(__file__).parent.parent.parent.parent.parent.parent / "xxxxxx/inference/"
        if self.infere_exp == True:
            self.path_infere_cfg = f'{self.save_path_inference}/{self.run_name}'
            os.makedirs(self.path_infere_cfg, exist_ok=True)
        
        print("training with cfg: ")
        print(self.config)
        if self.config['optimizer'] == "Adam":
            self.config['optimizer'] == "adam"


        if self.config['train_marglik']:
            print("Training with marginal likelihood")
           
            if margopt:
                if self.config['laplace']['laplace_type'] == "KronLaplace":
                    lap = KronLaplace
                elif self.config['laplace']['laplace_type'] == "DiagLaplace":
                    lap = DiagLaplace
                else:
                    raise ValueError("Invalid Laplace type!")
                if self.use_map:
                    la, model, margliks,val_perf = marglik_optimization(           
                        model=self.model, train_loader=self.train_loader,
                        valid_loader= self.valid_loader,likelihood=self.likelihood,
                        lr=self.config['lr'],
                        lr_min=self.config['laplace']['lr_min'],
                        n_epochs=self.config['num_epochs'],
                        optimizer=self.config['optimizer'],
                        laplace=lap,
                        prior_structure=self.config['laplace']['prior_structure'],
                        n_epochs_burnin= self.config['num_epochs']+1,
                        log_wandb = True,
                    )
                else:
                    
                    la, model, margliks, val_perf = marglik_optimization(           
                                model=self.model, train_loader=self.train_loader,
                                valid_loader= self.valid_loader,likelihood=self.likelihood,
                                lr=self.config['lr'],
                                lr_min=self.config['laplace']['lr_min'],
                                lr_hyp=self.config['laplace']['lr_hyp'],
                                n_epochs=self.config['num_epochs'],
                                optimizer=self.config['optimizer'],
                                laplace=lap,
                                prior_structure=self.config['laplace']['prior_structure'],
                                prior_prec_init=self.config['laplace']['prior_prec_init'],
                                n_epochs_burnin=self.config['laplace']['n_epochs_burnin'],
                                marglik_frequency=self.config['laplace']['marglik_frequency'],
                                n_hypersteps=self.config['laplace']['n_hypersteps'],
                                temperature= self.temperature,
                                target_sparsity=self.target_sparsity,
                                update_interval=self.update_interval,
                                sparse_method = self.sp_method,
                                log_wandb = True,
                            )
                
               
                             
    def finish_Wb(self):
        print("finishing logging")
        wandb.finish()     
      


                    
                

      

        

