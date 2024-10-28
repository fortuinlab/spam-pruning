from models import LeNet
from models_structured import PrunedLeNet
import numpy as np
import torch 
from matplotlib import pyplot as plt

from models import CancerNet_fc, LeNet, ResNet
from datasets_custom import CancerDataset, RotatedMNIST
from torch.nn.utils import prune
import torchvision.transforms as transforms
from laplace import Laplace 
from laplace import KronLaplace, DiagLaplace
import torch
import time 
from utils import evaluate_classification
from marglikopt import marglik_optimization
import os
import wandb

        
        
def get_unpruned_filter_indices(conv_layer):
    weights = conv_layer.weight.data.cpu().numpy()
    unpruned = np.any(weights != 0, axis=(1, 2, 3))  
    return np.nonzero(unpruned)[0]

def copy_weights(structured_zero_model, new_model, unpruned_indices):
    
    orig_layer = getattr(structured_zero_model, 'conv1')
    new_layer = getattr(new_model, 'conv1')
    indices = unpruned_indices['conv1']
    new_layer.weight.data = orig_layer.weight.data[indices, :, :, :]
    if orig_layer.bias is not None:
        new_layer.bias.data = orig_layer.bias.data[indices]

    
    for layer_name in ['conv2', 'conv3']:
        orig_layer = getattr(structured_zero_model, layer_name)
        new_layer = getattr(new_model, layer_name)
        out_indices = unpruned_indices[layer_name]

        
        if layer_name == 'conv2':
            in_indices = unpruned_indices['conv1']
        elif layer_name == 'conv3':
            in_indices = unpruned_indices['conv2']

        
        new_layer.weight.data = orig_layer.weight.data[out_indices, :, :, :][:, in_indices, :, :]
        if orig_layer.bias is not None:
            new_layer.bias.data = orig_layer.bias.data[out_indices]
            
    return new_model

if __name__ == """__main__""":
    #baseline_model = LeNet(n_out=10)
    #baseline_model.load_state_dict(torch.load('export_models_corr/LeNet_mnist_DiagLaplace_unitwise_100_wp_baseline_acc0.9650000333786011_marg_0.28357094526290894.pt'))

    structured_zero_model = LeNet(n_out=10)
    for model in os.listdir('/nfs/xxxxxx/pattern/LeNet_mnist_DiagLaplace_unitwise_100_wp_struct'):
        print(model)
        if model.endswith('.pt'):
    
            structured_zero_model.load_state_dict(torch.load('/nfs/xxxxxx/pattern/LeNet_mnist_DiagLaplace_unitwise_100_wp_struct/'+model))

            unpruned_indices = {
                'conv1': get_unpruned_filter_indices(structured_zero_model.conv1),
                'conv2': get_unpruned_filter_indices(structured_zero_model.conv2),
                'conv3': get_unpruned_filter_indices(structured_zero_model.conv3)
            }

            new_model = PrunedLeNet(
                n_filters_conv1=len(unpruned_indices['conv1']),
                n_filters_conv2=len(unpruned_indices['conv2']),
                n_filters_conv3=len(unpruned_indices['conv3'])
            )

            copy_weights(structured_zero_model, new_model, unpruned_indices)
            
            train_dataset = RotatedMNIST(root='./data', degree=0, train=True, download=True, transform=transforms.ToTensor())
            valid_dataset = RotatedMNIST(root='data', degree=0, train=False, download=True, transform=transforms.ToTensor())
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
            test_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=False)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(device)
            
            
            #torch.save(new_model.state_dict(), 'lenet_mnist_model_pruned.pt')
            #baseline_model.to(device)
            structured_zero_model.to(device)
            new_model.to(device)
            la, new_model, margliks, val_perf = marglik_optimization(           
                                    model=new_model, train_loader=train_loader,
                                    valid_loader= test_loader,likelihood="classification",
                                    lr=0.001,
                                    n_epochs=5,
                                    laplace=DiagLaplace,
                                    prior_structure="unitwise",
                                    log_wandb = False,
                                )

            print('marglik', margliks)
            print('val_perf', val_perf)
           
            
            



          

            





            #print('baseline model accuracy: {} %'.format(baseline_model_accuracy))
            #print('structured zero model accuracy: {} %'.format(structured_zero_model_accuracy))
            #print('pruned model accuracy: {} %'.format(pruned_model_accuracy))





    


