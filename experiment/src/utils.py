import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Tuple
import torch
from torch import nn, Tensor
import math
import numpy as np
from torchmetrics.functional import calibration_error
from torch.nn import functional as F


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



def check_sparsity(model):
    """Check the sparsity of a PyTorch model."""
    total_params = 0
    zero_params = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            total_params += param.numel()
            zero_params += torch.sum(param == 0).item()
    sparsity = zero_params / total_params
    return sparsity

def get_num_classes_from_model(model):

    for layer in reversed(model.children()):
        if isinstance(layer, nn.Linear):
            return layer.out_features

def evaluate_classification_detailed(new_model, test_loader, num_classes=2):
    device = next(new_model.parameters()).device
  
    new_model.eval()
    with torch.no_grad():
        nll = 0
        brier = 0
        correct = 0
        total = 0
        all_outputs = []  
        all_labels = []   
        print(len(test_loader))
        if len(test_loader.dataset[0]) == 2:
            for input, labels in test_loader:

                input = input.to(device)
                labels = labels.to(device)
                outputs = new_model(input)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                nll += nn.CrossEntropyLoss(reduction='sum')(outputs, labels).item()
                
                probs = F.softmax(outputs, dim=1)
                one_hot = F.one_hot(labels, num_classes=num_classes).float()
                brier += torch.sum((probs - one_hot) ** 2, dim=1).sum().item()

                all_outputs.append(outputs)
                all_labels.append(labels)

            all_outputs = torch.cat(all_outputs)
            all_labels = torch.cat(all_labels)
            
            #probs = F.softmax(all_outputs, dim=1)
            #one_hot = F.one_hot(all_labels, num_classes=num_classes).float()
            #brier = torch.sum((probs - one_hot) ** 2, dim=1).mean()

            ece = calibration_error(all_outputs, all_labels, n_bins=10, task='MULTICLASS', norm='l1',num_classes=num_classes).item()
            assert total == len(test_loader.dataset)
            return 100 * correct / total, nll / total, brier / total, ece
        else:
            return 0, 0, 0, 0
    
            
    


def evaluate_classification(new_model, test_loader):
    device = next(new_model.parameters()).device

    # Evaluate the model
    
    with torch.no_grad():
        correct = 0
        total = 0
        if len(test_loader.dataset[0]) == 2:
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = new_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        else: 
            for batch in test_loader:

                #print(batch.keys())
                #input_ids = batch['input_ids']
                #attention_mask = batch['attention_mask']
                labels = batch['labels']  # Fixed from 'labels' to 'label'
                # move to device
                #input_ids = input_ids.to(device)
                #attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                logits = new_model(batch)
                #loss = nn.CrossEntropyLoss()(logits, labels)
                #total_loss += loss.item()
                total += labels.size(0)
                correct += (logits.argmax(dim=1) == labels).sum().item()

        print('Accuracy of the network %d %%' % (
            100 * correct / total))
    return 100 * correct / total
    

def get_batch(source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape ``[full_seq_len, batch_size]``
        i: int

    Returns:
        tuple (data, target), where data has shape ``[seq_len, batch_size]`` and
        target has shape ``[seq_len * batch_size]``
    """
    seq_len = min(35, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

def evaluate_transformer(model, test_loader):
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for i in range(0, test_loader.size(0) - 1, 35):
            data, targets = get_batch(test_loader, i)
            device = next(model.parameters()).device
            targets = targets.to(device)
            data = data.to(device)
            seq_len = data.size(0)
            output = model(data)
            total_loss += seq_len * criterion(output, targets).item()
    ppl = math.exp(total_loss / (len(test_loader) - 1))
    return ppl 


