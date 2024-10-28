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
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import pandas as pd

def config_Wb(self):
    unique_id = wandb.util.generate_id()
    self.run_name = self.get_run_name()
    wandb.init(id = unique_id, name=self.run_name, project=self.config['wandb']['project'], entity=self.config['wandb']['entity'], config=to_log)
        
        
def get_unpruned_filter_indices(conv_layer):
    weights = conv_layer.weight.data.cpu().numpy()
    unpruned = np.any(weights != 0, axis=(1, 2, 3))  
    return np.nonzero(unpruned)[0]

def prune_conv_layer(model, layer_name, new_out_channels):
    # Get the original convolutional layer
    orig_layer = getattr(model, layer_name)
    
    # Create a new convolutional layer with the desired number of output channels
    new_layer = nn.Conv2d(orig_layer.in_channels, new_out_channels, kernel_size=orig_layer.kernel_size,
                          stride=orig_layer.stride, padding=orig_layer.padding, bias=orig_layer.bias is not None)
    
    # Copy the weights from the original layer to the new layer
    new_layer.weight.data = orig_layer.weight.data[:new_out_channels, :, :, :]
    if orig_layer.bias is not None:
        new_layer.bias.data = orig_layer.bias.data[:new_out_channels]
    
    # Replace the original layer in the model with the new one
    setattr(model, layer_name, new_layer)
    
def run_timed_inference(model, input_shape, batch_sizes, device, num_runs=10, warmup_runs=5):
    run_batch = { 'batch_size': batch_sizes, 'time_per_image': [], 'batch_time': []}

    for batch_size in batch_sizes:
        dummy_input = torch.randn((batch_size, *input_shape)).to(device)

       
        for _ in range(warmup_runs):
            model(dummy_input)

 
        torch.cuda.synchronize()
        start = time.time()

        for _ in range(num_runs):
            model(dummy_input)


        torch.cuda.synchronize()
        end = time.time()

        total_time = (end - start) * 1000 / num_runs  
        run_batch['time_per_image'].append(total_time / batch_size)
        run_batch['batch_time'].append(total_time)

    return run_batch

if __name__ == """__main__""":
    batch_time = []
    model_size = []
    sparsity_values = []
    val_ac_lst = []
    
    baseline_model = ResNet(depth=18, in_planes= 64,num_classes=10)
    baseline_model.load_state_dict(torch.load('/nfs/xxxxxx/pattern/ResNet_64_DiagLaplace_diagonal_100_wp_struct/ResNet_64_DiagLaplace_diagonal_100_wp_baseline_acc0.8267999291419983_marg_0.6696998476982117.pt'))
    sparsity_values.append(0)
    val_ac_lst.append(82.67999291419983)
    model_size.append(os.path.getsize('/nfs/xxxxxx/pattern/ResNet_64_DiagLaplace_diagonal_100_wp_struct/ResNet_64_DiagLaplace_diagonal_100_wp_baseline_acc0.8267999291419983_marg_0.6696998476982117.pt'))
    device = torch.device("cuda:0")
    baseline_model.to(device)
    print(device)
    run_batch = run_timed_inference(baseline_model, (3,32,32), [4,8,16,32,64,128,256,512], device)
    batch_time.append(run_batch['batch_time'])
    print(run_batch['batch_time'])
    
    
    structured_zero_model = ResNet(depth=18, in_planes= 64,num_classes=10)
    structured_zero_model.load_state_dict(torch.load('/nfs/xxxxxx/pattern/ResNet_64_DiagLaplace_diagonal_100_wp_struct/ResNet_64_DiagLaplace_diagonal_100_wp_acc0_marg_2.2487664222717285_sparsity_99_maskaftereach.pt'))
    new_resnet_model = ResNet(depth=18, in_planes= 64,num_classes=10)

 
    for name, module in structured_zero_model.named_modules():
        if isinstance(module, nn.Conv2d) and "conv1" not in name:
            # Get the unpruned channel indices from the corresponding layer in structured_zero_model
            unpruned_indices = get_unpruned_filter_indices(module)
            # Get the weights from the current layer
            weights = module.weight.data.cpu().numpy()

            # Create a new convolutional layer for the new_resnet_model with reduced channels
            new_conv_layer = nn.Conv2d(
                in_channels=len(unpruned_indices),  # Set the number of input channels
                out_channels=module.out_channels,  # Keep the same number of output channels
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                bias=module.bias is not None,
            )

            # Copy the weights corresponding to unpruned channels to the new layer
            new_conv_layer.weight.data = module.weight.data[unpruned_indices].clone()

            # If there is a bias, copy it as well
            if module.bias is not None:
                new_conv_layer.bias.data = module.bias.data.clone()

            # Add the new convolutional layer to the new_resnet_model
            setattr(new_resnet_model, name, new_conv_layer)

        elif isinstance(module, nn.Linear) and "fc" in name:
            # Handle the fully connected layer connected to the last convolutional layer
            new_linear_layer = nn.Linear(512, module.out_features, bias=module.bias is not None)
            new_linear_layer.weight.data = module.weight.data.clone()
            if module.bias is not None:
                new_linear_layer.bias.data = module.bias.data.clone()
            setattr(new_resnet_model, name, new_linear_layer)
            

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    tforms = [transforms.ToTensor(),
            transforms.Normalize(mean, std)]
    tforms_test = transforms.Compose(tforms)
    tforms_train = tforms_test
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=tforms_train)
    valid_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=tforms_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, )
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
    """"la, new_model, margliks, val_perf = marglik_optimization(           
                                    model=new_resnet_model, train_loader=train_loader,
                                    valid_loader= valid_loader,likelihood="classification",
                                    lr=0.001,
                                    n_epochs=5,
                                    laplace=DiagLaplace,
                                    optimizer = 'sgd',
                                    prior_structure="diagonal",
                                    log_wandb = False,
                                    )"""
    optimizer = torch.optim.SGD(new_resnet_model.parameters(), lr=0.000001, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0")
    new_resnet_model.to(device)
    new_resnet_model.train()
    for epoch in range(10):
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = new_resnet_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.8f}'.format(
                    epoch, batch_idx * len(inputs), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                
    val_acc = evaluate_classification(new_resnet_model, valid_loader)
    """print(val_acc)"""
    """la, new_model, margliks, val_perf = marglik_optimization(           
                                    model=new_resnet_model, train_loader=train_loader,
                                    valid_loader= valid_loader,likelihood="classification",
                                    lr=0.001,
                                    n_epochs=2,
                                    laplace=DiagLaplace,
                                    optimizer = 'sgd',
                                    prior_structure="diagonal",
                                    log_wandb = False,
                                    )"""
    
    """
    torch.save(new_resnet_model.state_dict(), 'ResNet_64_DiagLaplace_diagonal_99sp.pt')
    device = torch.device("cuda:0")
    new_resnet_model.to(device)
    sparsity_values.append(99)
    val_ac_lst.append(val_acc)
    batchsizes = [4,8,16,32,64,128,256,512]
    modelsz = os.path.getsize('ResNet_64_DiagLaplace_diagonal_99sp.pt')
    model_size.append(modelsz)
    print(modelsz)
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    run_batch = run_timed_inference(new_resnet_model, valid_loader.dataset[0][0].shape, batchsizes, device)
    batch_time.append(run_batch['batch_time'])
    print(run_batch['batch_time'])
    run_name = 'ResNet_64_DiagLaplace_diagonal'
    flattened_data = []
    for i, sparsity in enumerate(sparsity_values):
        for j, batch_size in enumerate(batchsizes):
            flattened_data.append({
                'Sparsity': sparsity,
                'Batch Size': batch_size,
                'Batch Time': batch_time[i][j]
            })
    df_flattened = pd.DataFrame(flattened_data)
    df = pd.DataFrame({
        'Model Size': model_size,
        "Val_Acc":val_ac_lst,
        'Sparsity': sparsity_values
    })

    # Sorting the DataFrame by 'Sparsity' values
    df_sorted = df.sort_values(by='Sparsity')
    df_flattened = pd.DataFrame(flattened_data)
    df_sorted_flt = df_flattened.sort_values(by='Sparsity')

    # Plotting with the sorted sparsity values
    plt.figure(figsize=(12, 8))
    ax1 = plt.gca()
    for batch_size in batchsizes:
        batch_data = df_sorted_flt[df_sorted_flt['Batch Size'] == batch_size]
        ax1.bar(batch_data['Sparsity'], batch_data['Batch Time'], label=f'Batch Size: {batch_size}')

    ax1.set_xlabel('Structure pruned (%)')
    ax1.set_ylabel('Batch Inference Time (ms)')
    ax1.legend(title='Batch Size')
    ax1.grid(True)

    ax2 = ax1.twinx()

    ax2.plot(df_sorted['Sparsity'], df_sorted['Val_Acc'], color='red', marker='x', label='Accuracy')
    ax2.set_ylabel('Accuracy (%)', color='red')
    for label in ax2.get_yticklabels():
        label.set_color('red')
    # Add legend to the secondary axis
    #ax2.legend(loc='upper right')
    ax2.legend(loc='upper left')

    plt.savefig(f"{run_name}_batch_time.pdf")
    plt.close()
    plt.figure(figsize=(10, 6))
    plt.plot(df_sorted['Sparsity'], df_sorted['Model Size'], marker='o')
    plt.xlabel('Sparsity (%)')
    plt.ylabel('Model Size (Byte)')
    plt.title('Model Size vs Sparsity')
    plt.grid(True)
    plt.savefig(f"{run_name}_model_size_{device}.pdf")
    plt.close()

    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    ax1.plot(df_sorted['Sparsity'], df_sorted['Model Size'], label='Model Memory size')
    ax1.set_xlabel('Sparsity')
    ax1.set_ylabel('Model Size (Byte)')
    ax1.legend()
    ax2 = ax1.twinx()
    ax2.plot(df_sorted['Sparsity'], df_sorted['Val_Acc'], color='red', marker='x', label='Accuracy')
    ax2.set_ylabel('Accuracy (%)', color='red')
    for label in ax2.get_yticklabels():
        label.set_color('red')
    # Add legend to the secondary axis
    #ax2.legend(loc='upper right')
    ax2.legend(loc='upper left')

    plt.title('Model Size vs Validation Accuracy')
    plt.grid(True)
    plt.savefig(f"{run_name}_model_size_acc_bat_{device}.pdf")
    plt.close()
    # make a bar plot with sparsity vs batch time as we have only 2 models
    plt.figure(figsize=(12, 8))
    ax1 = plt.gca()
    for batch_size in batchsizes:
        batch_data = df_sorted_flt[df_sorted_flt['Batch Size'] == batch_size]
        ax1.bar(batch_data['Sparsity'], batch_data['Batch Time'], label=f'Batch Size: {batch_size}')
        
    ax1.set_xlabel('Structure pruned (%)')
    ax1.set_ylabel('Batch Inference Time (ms)')
    ax1.legend(title='Batch Size')
    ax1.grid(True)
    plt.savefig(f"{run_name}_bar_batch_time.pdf")
    plt.close()
    
    """
    
            

            
        
    
    
    
            
            



          

            










    


