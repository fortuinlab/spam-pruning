
import time
import pandas as pd
import pickle
import os
import numpy as np
#import matplotlib.pyplot as plt

import wandb
import copy
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets 


from models_DEBUG import (
    CancerNet_fc,
    BostonReg_fc,
    ResNet18,
    MNIST_FC,
    Cifar10_fc,
    Cifar10_CNN,
    ResNet,
    WideResNet,
    LanguageTransformer,
    #VisionTransformer,
)

from datasets_DEBUG import (
    CancerDataset,
    CancerDataset_supported,
    BostonHousingDataset,
    CIFAR10Dataset,
    MNISTDataset,
    RotatedCIFAR100,
    WikiText2Dataset,
)

from sparsify_v2 import (
    sparse_strategy,
    magnitude_pruning,
    random_strategy,
    SNIP_strategy,
    GraSP_strategy,
    #laplace_abs,
)

from utils import check_sparsity


import sys 
sys.path.append('..')
#from Laplace.laplace import Laplace
#from Laplace.laplace.curvature.backpack import BackPackGGN
#from Laplace.laplace.curvature.asdl import AsdlGGN, AsdlEF, AsdlHessian, AsdlInterface

from laplace import KronLaplace, DiagLaplace

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

        self.resnet_inplanes = self.config.get('resnet_inplanes', 32)
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
            
            """if laplace_type is not None:
                if sparse_method is not None:
                    return f'{model_name}_{laplace_type}_{prior}_{epoch}_Sparse_{sparse_method}'
                else:
                    if self.use_map:
                        return f'{model_name}_{laplace_type}_{prior}_{epoch}_map_new'
                    else:
                        
                        if self.resnet_inplanes == 64 and self.config['model_name'] == 'ResNet':
                            return f'{model_name}_64_{laplace_type}_{prior}_{epoch}_wp'
                        else:
                            return f'{model_name}_{laplace_type}_{prior}_{epoch}_wp'
                        
            else:
                if sparse_method is not None:
                    return f'{model_name}_{prior}_{epoch}_{sparse_method}'
                else:
                    return f'{model_name}_{prior}_{epoch}' """
            return f'{model_name}_{epoch}_DEBUG'  
        else:
            if epoch is not None:
                return f'{model_name}_{epoch}'
            else:
                return model_name
            
    def config_Wb(self):
        unique_id = wandb.util.generate_id()
        self.run_name = self.get_run_name()
        # excluding the laplace part from the config as we are using the new maglikopt script
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
            if self.config['model_name'] == 'ResNet':
                    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
                    std = [x / 255 for x in [63.0, 62.1, 66.7]]

                    tforms = [transforms.ToTensor(),
                            transforms.Normalize(mean, std)]
                    tforms_test = transforms.Compose(tforms)
                    tforms_train = tforms_test
                    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=tforms_train)
                    valid_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=tforms_test)

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

        elif dataset_name == "mnist":
            self.likelihood = "classification"
            train_dataset = MNISTDataset(root='data', train=True, transform=transforms.ToTensor())
            valid_dataset = MNISTDataset(root='data', train=False, transform=transforms.ToTensor())
        
        elif dataset_name == "WikiText2":
            train_dataset = WikiText2Dataset(split='train')
            valid_dataset = WikiText2Dataset(split='val',batch_size=10)
            self.vocab_size = len(train_dataset.vocab)
            self.likelihood = "classification"
            self.train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)
            self.valid_loader = DataLoader(valid_dataset, batch_size=None, shuffle=False)
        else:
            raise ValueError("Invalid dataset name!")
 
                                
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

        elif model_name == "LanguageTransformer":
            # model hyperparameters
            emsize = 200  # embedding dimension
            d_hid = 200  # dimension of the feedforward network model in ``nn.TransformerEncoder``
            nlayers = 2  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
            nhead = 2  # number of heads in ``nn.MultiheadAttention``
            dropout = 0.2  # dropout probability
            # pass variables explicitly
            print("Vocab size: ", self.vocab_size)
            print("size loader", len(self.train_loader))
            self.model = LanguageTransformer(ntoken=self.vocab_size, d_model=emsize, nhead=nhead, d_hid=d_hid, nlayers=nlayers, dropout=dropout)
        
        elif model_name == "VisionTransformer":
            self.model = VisionTransformer()
        
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
        # this will be in the main.py file after I get it working
        # add the
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
                     la, model, margliks = marglik_optimization(           
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
                    
                        la, model, margliks = marglik_optimization(           
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
                                    n_epochs_burnin=0, #self.config['laplace']['n_epochs_burnin'], # just to quickly use MAP
                                    marglik_frequency=self.config['laplace']['marglik_frequency'],
                                    n_hypersteps=self.config['laplace']['n_hypersteps'],

                                    log_wandb = True,
                                )
                sp = check_sparsity(model)
                print(f"Sparsity: {sp}")
                sparsities = [20,40,60,70,75,80,85,90,95,99]
                #TODO: more sparsities 
                sparse_list = {'laplacekron':{'model': copy.deepcopy(model), 'function': sparse_strategy ,'sparsities': sparsities, 'la': la},
                                #'lp_posterior_abs':{'model': copy.deepcopy(model), 'function': laplace_abs ,'sparsities': sparsities, 'la': la},
                                'magnitude':{'model': copy.deepcopy(model), 'function': magnitude_pruning ,'sparsities': sparsities, 'la': la},
                                'random':{'model': copy.deepcopy(model), 'function': random_strategy ,'sparsities': sparsities, 'la': la},
                                'SNIP':{'model': copy.deepcopy(model), 'function': SNIP_strategy ,'sparsities': sparsities, 'la': la},
                                'GraSP':{'model': copy.deepcopy(model), 'function': GraSP_strategy ,'sparsities': sparsities, 'la': la},
                                }
              
                if self.sparse_exact_same == True:
                    for sparse_name, sparse_dict in sparse_list.items():
                        with wandb.init(reinit=True, id=wandb.util.generate_id(), project=self.config['wandb']['project'], entity=self.config['wandb']['entity'], name=f"{self.run_name}_sub_{sparse_name}"): #_fine_tune"):
                            #try:
                            sparse_dict['function'](la=sparse_dict['la'],model = sparse_dict['model'], test_loader= self.valid_loader,
                                                     train_loader = self.train_loader, sparsities = sparse_dict['sparsities'])
                            #except:
                            #    print(f"Sparse strategy {sparse_name} failed")
                            #    continue
                              
                else:
                    if self.sparse_method == "laplace":
                        sparse_strategy(la, model, self.valid_loader, sparsities)
                    elif self.sparse_method == "magnitude":
                        magnitude_pruning(la, model, self.valid_loader, sparsities)
                    elif self.sparse_method == "random":
                        random_strategy(la, model, self.valid_loader, sparsities)
                    else:
                        print("No sparse method provided. Skipping sparse strategy.")


                if self.pickle_laplace == True:

                    os.makedirs(f'./results/{self.run_name}', exist_ok=True)
                    with open(f'./results/{self.run_name}/laplace.pkl', 'wb') as f:
                        pickle.dump(la, f)
                    #with open(f'./results/{self.run_name}/model.pkl', 'wb') as f:
                    #    pickle.dump(model, f)
                    with open(f'./results/{self.run_name}/margliks.pkl', 'wb') as f:
                        pickle.dump(margliks, f)

                    torch.save(model.state_dict(),(f'./results/{self.run_name}/model.pt'))
            print("Finished training with marginal likelihood")

    
                             
    
    def finish_Wb(self):
        print("finishing logging")
        wandb.finish()     
      


                    
                

      

        

