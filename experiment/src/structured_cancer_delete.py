import numpy as np
import matplotlib.pyplot as plt
from models_structured import PrunedCancerNet_fc, PrunedLeNet
from models import LeNet, CancerNet_fc
from datasets_custom import CancerDataset, RotatedMNIST
from laplace import marglik_training
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
import time
import wandb
import pandas as pd
from marglikopt import marglik_optimization
from laplace import KronLaplace, DiagLaplace
import shutil

def run_timed_inference(model, input_shape, batch_sizes, device, num_runs=10, warmup_runs=5):
    run_batch = { 'batch_size': batch_sizes, 'time_per_image': [], 'batch_time': []}

    for batch_size in batch_sizes:
        dummy_input = torch.randn((batch_size, *input_shape)).to(device)

        # Warm-up runs
        for _ in range(warmup_runs):
            model(dummy_input)

        #torch.cuda.synchronize()  
        torch.cuda.synchronize()
        start = time.time()
        
        # Multiple runs for averaging
        for _ in range(num_runs):
            model(dummy_input)

        #torch.cuda.synchronize()  # Synchronize after runs
        torch.cuda.synchronize()
        end = time.time()

        total_time = (end - start) * 1000 / num_runs  # Convert to milliseconds
        run_batch['time_per_image'].append(total_time / batch_size)
        run_batch['batch_time'].append(total_time)

    return run_batch


def get_unpruned_neuron_indices(fc_layer):
    weights = fc_layer.weight.data.cpu().numpy()
    unpruned = np.any(weights != 0, axis=0)  
    return np.nonzero(unpruned)[0]

def copy_weights_fc(structured_zero_model, new_model, unpruned_indices):
    # Handle the first layer separately
    orig_layer = getattr(structured_zero_model, 'fc1')
    new_layer = getattr(new_model, 'fc1')
    indices = unpruned_indices['fc1']
    new_layer.weight.data = orig_layer.weight.data[indices, :]
    if orig_layer.bias is not None:
        new_layer.bias.data = orig_layer.bias.data[indices]

    # Handle subsequent layers
    for layer_name in ['fc2', 'fc3']:  # Adjust as per your model's layer names
        if layer_name not in unpruned_indices:
            continue  # Skip if layer is not in the unpruned indices

        orig_layer = getattr(structured_zero_model, layer_name)
        new_layer = getattr(new_model, layer_name)
        out_indices = unpruned_indices[layer_name]
        
        # Get input indices from the previous layer's unpruned indices
        if layer_name == 'fc2':
            in_indices = unpruned_indices['fc1']
        elif layer_name == 'fc3':
            in_indices = unpruned_indices['fc2']

        # Copy weights considering both input and output indices
        new_layer.weight.data = orig_layer.weight.data[out_indices, :][:, in_indices]
        if orig_layer.bias is not None:
            new_layer.bias.data = orig_layer.bias.data[out_indices]


def evaluate_classification(model, loader, device):
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


if __name__ == "__main__":
    structured_zero_model = CancerNet_fc(30,100,2)

    train_dataset = CancerDataset(train=True)
    valid_dataset = CancerDataset(train=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=False)

    dir_zeromod = '/nfs/xxxxxx/pattern/CancerNet_fc_breast_cancer_DiagLaplace_scalar_50_wp_struct'
    removed_strture = dir_zeromod + '/removed_structure_new'
    os.makedirs(removed_strture,exist_ok=True)
    if os.path.exists(removed_strture) and os.listdir(removed_strture):
    # Get a list of all pruned model files in the directory
        pruned_model_files = [f for f in os.listdir(removed_strture) if f.endswith('.pt')]
    
    else:
        pruned_model_files = []
      
        
    
    models = [f for f in os.listdir(dir_zeromod) if f.endswith('.pt')] 

    sparsity_models = {}
    for filename in models:
        sparsity_level = filename.split('_')[-2]
        if sparsity_level not in sparsity_models:
            sparsity_models[sparsity_level] = filename

    # List of unique models, one for each sparsity level
    unique_models_per_sparsity = list(sparsity_models.values())
    models = sparsity_models
    device = torch.device('cuda:0')
    sparsity_values = [] 
    batch_time = []
    batchsizes = [4,8,16,32,64,128,256,512]
    model_size = []
    val_ac_lst = []
    margliks_lst = []
    splits = models["99"].split("_")
    run_name = "_".join(splits[:6]) + "_compressed_meta"
    
    wandb_log = False
    train = False
    print(sparsity_models)
    if wandb_log:
        unique_id = wandb.util.generate_id()
        wandb.init(id = unique_id, name=run_name, project="BNN_Sparse", entity="xxxxxx")
      
    
    
    found_matching_model = False
    for sparsity_level, filename in sparsity_models.items():
        if "baseline" in filename:
            model_sparsity = 0
            model = CancerNet_fc(30,100,2)
            model.load_state_dict(torch.load(os.path.join(dir_zeromod, filename)))
            device= "cpu"
            model.to(device)
            val_acc = evaluate_classification(model, test_loader,device)
            run_batch = run_timed_inference(model, test_loader.dataset[0][0].shape, batchsizes, device)
            print(f"Model: {model}, Sparsity: {model_sparsity}, Val Acc: {val_acc}")
            print(run_batch)
            print("-----------")
            sparsity_values.append(int(model_sparsity))
            batch_time.append(run_batch["batch_time"])
            mod_size= os.path.getsize(os.path.join(dir_zeromod, filename))
            
            model_iterm = filename.replace(".pt", "")
            marg = float(model_iterm.split("_")[-1])
            if wandb_log:
                wandb.log({"sparsity": model_sparsity, "val_acc": val_acc,
                            "run_batch": run_batch, "model_size": mod_size,"marglik":marg})
                
            margliks_lst.append(marg)
            model_size.append(mod_size)
            val_ac_lst.append(val_acc)
                

        else:
            model_sparsity = int(sparsity_level)
            for model in pruned_model_files:
                if str(model_sparsity) in model:
                    found_matching_model = True
                    print("model already pruned")
                    model_meta = torch.load(os.path.join(removed_strture, model))['model_meta']
                    new_model = PrunedCancerNet_fc(**model_meta)
                    new_model.load_state_dict(torch.load(os.path.join(removed_strture, model))['model_state_dict'])
                    splits = model.replace(".pt", "").split("_")
                    sparsity = int(splits[4])
                    accuracy = float(splits[splits.index("acc") + 1])
                    marg = float(splits[splits.index("marg") + 1])
                    new_model.to(device)
                    run_batch = run_timed_inference(new_model, test_loader.dataset[0][0].shape, batchsizes, device)
                    mod_size = os.path.getsize(os.path.join(removed_strture, model))
                    model_size.append(mod_size)
                    batch_time.append(run_batch["batch_time"])
                    val_ac_lst.append(accuracy)
                    margliks_lst.append(marg)
                    sparsity_values.append(sparsity)
          
                
        if not found_matching_model:   
            structured_zero_model = CancerNet_fc(30,100,2)
            structured_zero_model.load_state_dict(torch.load(os.path.join(dir_zeromod, filename)))
            #print(structured_zero_model)
            unpruned_indices_fc1 = get_unpruned_neuron_indices(structured_zero_model.fc1)
            unpruned_indices_fc2 = get_unpruned_neuron_indices(structured_zero_model.fc2)
            new_model = PrunedCancerNet_fc(input_size = 30, hidden_size1 = len(unpruned_indices_fc1), hidden_size2 = len(unpruned_indices_fc2), output_size = 2)
            copy_weights_fc(structured_zero_model, new_model, {'fc1': unpruned_indices_fc1, 'fc2': unpruned_indices_fc2})
            print(new_model)
            train = True
            if train:
                if "Kron" in filename:
                    lap = KronLaplace
                else:
                    lap = DiagLaplace
                prior_structure = filename.split("_")[5]
                la, new_model, margliks, val_perf = marglik_optimization(           
                                    model=new_model, train_loader=train_loader,
                                    valid_loader= test_loader,likelihood="classification",
                                    lr=0.001,
                                    n_epochs=10,
                                    laplace=lap,
                                    prior_structure=prior_structure,
                                    log_wandb = False,
                            )
            val_ac_lst.append(val_perf*100)
            margliks_lst.append(margliks[-1])
            sparsity_values.append(model_sparsity)

            device = "cpu"
            new_model.to(device)
            run_batch = run_timed_inference(new_model, test_loader.dataset[0][0].shape, batchsizes, device)
            splits = filename.split("_") 
            model_name_new = "_".join(splits[:4]) + f"reducued_{model_sparsity}_acc_{val_perf*100}_marg_{margliks[-1]}.pt"
            model_meta = {
                    "hidden_size1": len(unpruned_indices_fc1),
                    "hidden_size2": len(unpruned_indices_fc2),
                    "input_size": 30,
                    "output_size": 2
            }
            # save model with meta data
            torch.save({
                    'model_state_dict': new_model.state_dict(),
                    'model_meta': model_meta,
            }, os.path.join(removed_strture, model_name_new))
            
            mod_size = os.path.getsize(os.path.join(removed_strture, model_name_new))
            model_size.append(mod_size)
            batch_time.append(run_batch["batch_time"])
            if wandb_log:
                wandb.log({"sparsity": model_sparsity, "val_acc": val_perf*100,
                    "run_batch": run_batch, "model_size": mod_size,"marglik":margliks[-1]})

    
    print(sparsity_values)
    print(model_size)
    print(val_ac_lst)
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
        ax1.plot(batch_data['Sparsity'], batch_data['Batch Time'], marker='o', label=f'Batch Size: {batch_size}')

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
    plt.savefig(f"{run_name}_model_size.pdf")
    plt.close()

    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    ax1.plot(df_sorted['Sparsity'], df_sorted['Model Size'], marker='o', label='Model Memory size')
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
    plt.savefig(f"{run_name}_model_size_acc.pdf")
    plt.close()

