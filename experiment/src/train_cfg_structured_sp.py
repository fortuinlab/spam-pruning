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
from matplotlib import pyplot as plt

# own  modules
from models import (CancerNet_fc, BostonReg_fc, ResNet18, MNIST_FC, Cifar10_fc, 
                    Cifar10_CNN, ResNet, WideResNet, LanguageTransformer, LeNet, VisionTransformer, MLPMixer)
from datasets_custom import (CancerDataset, CancerDataset_supported, BostonHousingDataset, 
                      CIFAR10Dataset, MNISTDataset, RotatedCIFAR100, RotatedMNIST)
from utils import check_sparsity, evaluate_classification

from sparsify_v2 import (sparse_strategy, magnitude_pruning, random_strategy, 
                         SNIP_strategy, GraSP_strategy, laplace_abs)
from structured_sparsity import prune_neurons_based_on_precision, apply_masks
from marglikopt import marglik_optimization

import logging
logging.basicConfig(level=logging.INFO)

# seeds 

seed = np.random.randint(0, 200)
torch.manual_seed(seed)
np.random.seed(seed)



margopt = True

class train_cfg():
    def __init__(self, config, cfg_name):
        self.config = config
        self.cfg_name = cfg_name
        self.run_name = None  # initialize to None
        self.pickle_laplace = False
        self.sparse_method = config.get('sparse_method', None)
        self.sparse_exact_same = True
        self.use_map = self.config.get('use_map', False)
        self.infere_exp = True
        self.resnet_inplanes = self.config.get('resnet_inplanes', 32)
        # patter_only export in case we skipped one export and only interested in the pattern
        self.pattern_only_export = False

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
            
            if laplace_type is not None:
                if sparse_method is not None:
                    return f'{model_name}_{laplace_type}_{prior}_{epoch}_Sparse_{sparse_method}'
                else:
                    if self.use_map or self.config['laplace']['n_epochs_burnin']> epoch :
                        self.use_map = True
                        return f'{model_name}_{self.config["dataset_name"]}_{epoch}_MAP'
                    else:
                        if self.resnet_inplanes == 64 and self.config['model_name'] == 'ResNet':
                            return f'{model_name}_64_{laplace_type}_{prior}_{epoch}_wp'
                        else:
                            return f'{model_name}_{self.config["dataset_name"]}_{laplace_type}_{prior}_{epoch}_wp'
                        
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
        elif dataset_name == "Cifar100":
            self.likelihood = "classification"
            tforms = [transforms.ToTensor(),    # first, convert image to PyTorch tensor
                    transforms.Normalize((0.1307,), (0.3081,))]
            
            tforms_test = transforms.Compose(tforms)
            tforms_train = tforms_test
            train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=tforms_train)
            valid_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=tforms_test)

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
                self.model = WideResNet(depth=28, widen_factor=10, num_classes=10)
            elif dataset_name == "Cifar100":
                self.model = WideResNet(depth=28, widen_factor=10, num_classes=100)


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
         
        self.save_path_inference = Path(__file__).parent.parent.parent.parent.parent.parent / "xxxxxx/pattern/"
        self.path_infere_cfg = f'{self.save_path_inference}/{self.run_name}_struct/'
        if self.infere_exp == True:
            os.makedirs(self.path_infere_cfg, exist_ok=True)
        
        print("training with cfg: ")
        print(self.config)
        if self.config['optimizer'] == "Adam":
            self.config['optimizer'] == "adam"

   
        
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
                valid_loader=self.valid_loader, likelihood=self.likelihood,
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
                log_wandb=True,
            )

        sp = check_sparsity(model)
        print(f"Sparsity: {sp}")
        sparsities = [20, 40, 60, 70, 75, 80, 85, 90, 95, 99]
        
        model_name = self.config['model_name']
        if model_name == "ResNet":
            args_sparse = { 
                'prior_structure': self.config['laplace']['prior_structure'],
                'tune_epochs_burnin': 11 if self.use_map else 5,
                "lr": 0.1,
                "lr_hyp": 0.1,
                "lr_min": 1e-06,
                "marglik_frequency": 1,
                "n_hypersteps": 50,
                'fine_tune': self.config["tune"],
                'tune_epochs': 10,
                'lr': self.config['lr']

            }
        
        else:
            args_sparse = {
                'prior_structure': self.config['laplace']['prior_structure'],
                'tune_epochs_burnin': 11 if self.use_map else 0,
                'marglik_frequency': self.config['laplace']['marglik_frequency'],
                'fine_tune': self.config["tune"],
                'tune_epochs': 10,
                'lr': self.config['lr']*0.1
            }

        cfgs = self.config
        run_name = f"{self.run_name}_structured_pruning_wp_maskaftereach"
        
        model_baseline_name = self.path_infere_cfg + f"/{self.run_name}_baseline_acc{val_perf}_marg_{margliks[-1]}.pt"
        torch.save(model.state_dict(), model_baseline_name)
        
        weights = [param.view(-1) for param in model.parameters()]
        weights_sq = [w ** 2 for w in weights]
        weights_sq_flat = torch.cat(weights_sq)

        if isinstance(la, KronLaplace):
            precision_values = la.posterior_precision.diag().detach().cpu().numpy()
        else:
            precision_values = la.posterior_precision.detach().cpu().numpy()
        p_w = precision_values * weights_sq_flat.detach().cpu().numpy()

        with wandb.init(reinit=True, id=wandb.util.generate_id(), config=cfgs, project=self.config['wandb']['project'], entity=self.config['wandb']['entity'], name=run_name):
            wandb.config.update(args_sparse, allow_val_change=True)
            wandb.config.update({'sparsification_method': "structured"}, allow_val_change=True)
            for i in range(len(sparsities)):
                model_copy = copy.deepcopy(model)
                mask_dict = {}
                total_neurons_pruned = 0
                total_filters_pruned = 0
                precision_index = 0
                prune_percentage = sparsities[i] / 100
                for name, module in model_copy.named_modules():
                    if isinstance(module, (nn.Linear, nn.Conv2d)):
                        precision_index, num_pruned = prune_neurons_based_on_precision(module, p_w, precision_index, prune_percentage)
                    if isinstance(module, nn.Linear):
                        total_neurons_pruned += num_pruned
                    elif isinstance(module, nn.Conv2d):
                        total_filters_pruned += num_pruned

                print('Total neurons pruned:', total_neurons_pruned)
                print('Total filters pruned:', total_filters_pruned)
                mask_dict = {}
                for name, module in model_copy.named_modules():
                    if model_name == "ResNet" and isinstance(module, nn.Conv2d):
                        mask = module.weight.data != 0
                        mask_dict[name + '.weight'] = mask
                    else :         
                        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                            mask = module.weight.data != 0
                            mask_dict[name + '.weight'] = mask
                if model_name == "ResNet":
                    la, model_copy, margliks = marglik_optimization(
                        model=model_copy, train_loader=self.train_loader,
                        valid_loader=None, likelihood=self.likelihood,
                        lr=args_sparse['lr'],
                        lr_min=self.config['laplace']['lr_min'],
                        lr_hyp=self.config['laplace']['lr_hyp'],
                        n_epochs=args_sparse["tune_epochs"],
                        n_hypersteps=args_sparse["n_hypersteps"],
                        optimizer=self.config['optimizer'],
                        laplace=lap,
                        prior_structure=self.config['laplace']['prior_structure'],
                        n_epochs_burnin=args_sparse["tune_epochs_burnin"],
                        log_wandb=True,
                        mask_dict=mask_dict,
                    )
                else:
                    la, model_copy, margliks = marglik_optimization(
                        model=model_copy, train_loader=self.train_loader,
                        valid_loader=None, likelihood=self.likelihood,
                        lr=args_sparse['lr'],
                        #lr_min=self.config['laplace']['lr_min'],
                        #lr_hyp=self.config['laplace']['lr_hyp'],
                        n_epochs=args_sparse["tune_epochs"],
                        optimizer=self.config['optimizer'],
                        laplace=lap,
                        prior_structure=self.config['laplace']['prior_structure'],
                        n_epochs_burnin=args_sparse["tune_epochs_burnin"],
                        log_wandb=True,
                        mask_dict=mask_dict,
                    )
                val_acc = evaluate_classification(model_copy, self.valid_loader)
                wandb.log({"sparsity": sparsities[i], "val_acc": val_acc, "marglik": margliks[-1], "total_neurons_pruned": total_neurons_pruned, "total_filters_pruned": total_filters_pruned})

                if self.infere_exp == True:
                    model_name = self.path_infere_cfg + f"{self.run_name}_acc_{val_acc}_marg_{margliks[-1]}_sparsity_{sparsities[i]}_maskaftereach.pt"
                    torch.save(model_copy.state_dict(), model_name)
                    """try:
                        plt.figure(figsize=(10, 10))
                        for name, module in model_copy.named_modules():
                            if isinstance(module, nn.Linear):
                                weights = module.weight.data.flatten().cpu().numpy()
                                plt.hist(weights, bins=50, alpha=0.5, label=name)
                        plt.legend()
                        plt.xlabel('Weight')
                        plt.ylabel('Frequency')
                        plt.title('Distribution of Model Weights')
                        plt.savefig(f'fig/{self.run_name}_weights_distribution_{sparsities[i]}.pdf')                    
                        # show the weight matrix after pruning using matplotlib
                        plt.figure(figsize=(10, 10))
                        for name, module in model_copy.named_modules():
                            if isinstance(module, nn.Linear):
                                weights = module.weight.data.cpu().numpy()
                                plt.imshow(weights, cmap='viridis')
                                plt.colorbar()
                                plt.title(name)
                                plt.savefig(f'fig/{self.run_name}_weightmat_{name}_{sparsities[i]}.pdf')  

                        for name, param in model_copy.named_parameters():
                            wandb.log({name: wandb.Histogram(param.detach().cpu().numpy())})
                    except:
                        pass"""
                        
                     
                                                   

                
                

        if self.pickle_laplace == True:

            os.makedirs(f'./results/{self.run_name}', exist_ok=True)
            with open(f'./results/{self.run_name}/laplace.pkl', 'wb') as f:
                pickle.dump(la, f)
            with open(f'./results/{self.run_name}/model.pkl', 'wb') as f:
                pickle.dump(model, f)
            with open(f'./results/{self.run_name}/margliks.pkl', 'wb') as f:
                pickle.dump(margliks, f)

            torch.save(model.state_dict(),(f'./results/{self.run_name}/model.pt'))
            
    print("Finished training with marginal likelihood")

        
                             
    def finish_Wb(self):
        print("finishing logging")
        wandb.finish()     
      


                    
                

      

        

