from asdfghjkl.operations import Bias, Scale
import torch 
import torch.nn as nn

"""def apply_masks(model, mask_dict):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            mask = mask_dict.get(name + '.weight')
            if mask is not None:
                module.weight.data *= mask"""

def create_mask_from_scores(scores, prune_percentage):
    """
    Create a binary mask from the given scores based on the specified pruning percentage.

    Parameters:
    scores (Tensor): Scores for each parameter in the model.
    prune_percentage (float): The percentage of parameters to prune.

    Returns:
    Tensor: A binary mask indicating which parameters to keep (1) or prune (0).
    """
    num_params_to_keep = int((1.0 - prune_percentage) * scores.numel())
    threshold, _ = torch.kthvalue(scores, num_params_to_keep)
    mask = scores >= threshold
    return mask

def structured_prune(model, scores, prune_percentage):
    mask_dict = {}
    total_neurons_pruned = 0
    total_filters_pruned = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            weight_score_name = name + '.weight'
            if weight_score_name in scores:
                score = scores[weight_score_name]

                if isinstance(module, nn.Conv2d):
                    aggregated_scores = score.sum(dim=[1, 2, 3])
                else:  # nn.Linear
                    aggregated_scores = score.sum(dim=1)

                num_elements = aggregated_scores.numel()
                num_to_prune = min(int(prune_percentage * num_elements), num_elements)

                if num_to_prune > 0:
                    threshold, _ = torch.topk(aggregated_scores, num_to_prune, largest=False)
                    threshold_value = threshold[-1].item()
                    zero_mask = aggregated_scores <= threshold_value
                else:
                    zero_mask = torch.zeros_like(aggregated_scores, dtype=torch.bool)

                for i, mask in enumerate(zero_mask):
                    if mask:
                        if isinstance(module, nn.Conv2d):
                            module.weight.data[i] = 0
                            if module.bias is not None:
                                module.bias.data[i] = 0
                            total_filters_pruned += 1
                        else:  # nn.Linear
                            if module.out_features not in [10, 2]:  # Skip output layer with 10 or 2 neurons
                                module.weight.data[i] = 0
                                total_neurons_pruned += 1

    return mask_dict, total_neurons_pruned, total_filters_pruned











@torch.no_grad()            
def linearize(model):
    """linearizes the model"""
    signs = {}
    for name, param in model.state_dict().items():
        signs[name] = torch.sign(param)
        param.abs_()
    return signs
@torch.no_grad()
def nonlinearize(model, signs):
    """nonlinearizes the model"""
    for name, param in model.state_dict().items():
        param.mul_(signs[name])
        

def synflow(model, loss, dataloader, device):
    """returns the synflow scores for the model"""
    signs = linearize(model)
    (data, _) = next(iter(dataloader))
    input_dim = list(data[0,:].shape)
    input = torch.ones([1] + input_dim).to(device)#, dtype=torch.float64).to(device)
    output = model(input)
    torch.sum(output).backward()
    nonlinearize(model, signs)
    return model.scores


        
        

 
def apply_structured_mask(pruner, sparsity, device, model):
    scores_list = [
        pruner.scores.get(id(param), torch.tensor([], device=device)).to(device)
        for _, param in pruner.masked_parameters
        if id(param) in pruner.scores
    ]

    # Check if there are any scores to concatenate
  

    global_scores = torch.cat(scores_list)

    # Compute the threshold to determine which parameters to prune
    k = int((1.0 - sparsity) * global_scores.numel())
    threshold, _ = torch.kthvalue(global_scores, k)

    # Apply masks based on scores
    for mask, param in pruner.masked_parameters:
        param_id = id(param)
        if param_id in pruner.scores:
            score = pruner.scores[param_id]
            zero_mask = score <= threshold
            if len(param.size()) == 4:  # Conv2d layers
                # Expand the mask for Conv2d filters
                new_mask = zero_mask[:, None, None, None].detach()
                mask.data.copy_(new_mask)
            elif len(param.size()) == 2:  # Linear layers
                # Expand the mask for Linear layers' neurons
                new_mask = zero_mask[:, None].detach()
                mask.data.copy_(new_mask)
            elif len(param.size()) == 1:  # Biases or 1D parameters
                # Apply mask directly for 1D parameters like biases
                new_mask = zero_mask.detach()
                mask.data.copy_(new_mask)
        else:
            print(f"No score found for param ID: {param_id}")
    pruner.apply_mask()

 
 
                
def apply_masks(model, mask_dict):
    for name, module in model.named_modules():
        mask = mask_dict.get(name + '.weight')
        if mask is not None and hasattr(module, 'weight'):
            module.weight.data *= mask


def prune_neurons_based_on_precision(module, precision_values, precision_index, prune_percentage, prune_rows=False):
    """ Prune a percentage of rows or columns based on posterior precision. """

    layer_precision_count = 0
    num_pruned = 0
    if isinstance(module, nn.Linear):

        if module.weight.data.shape[0] == 2 or module.weight.data.shape[0] == 10:
            return 0, 0

        num_rows, num_cols = module.weight.data.shape

        layer_precision_count = num_rows * num_cols

        layer_precision_values = precision_values[precision_index:precision_index + layer_precision_count]
        reshaped_precision = layer_precision_values.reshape(num_rows, num_cols)

        if prune_rows:
            sum_precision = reshaped_precision.sum(axis=1)
        else:
            sum_precision = reshaped_precision.sum(axis=0)
        num_to_prune = int(prune_percentage * (num_rows if prune_rows else num_cols))

        _, indices_to_prune = torch.topk(sum_precision, num_to_prune, largest=False)

        for index in indices_to_prune:
            if prune_rows:
                module.weight.data[index] = 0
            else:
                module.weight.data[:, index] = 0

        num_pruned += len(indices_to_prune)

        precision_index += layer_precision_count

        return precision_index, num_pruned
    elif isinstance(module, nn.Conv2d):
        
        out_channels, in_channels, kernel_height, kernel_width = module.weight.data.shape
        layer_precision_count = out_channels * in_channels * kernel_height * kernel_width

        reshaped_precision = precision_values[precision_index:precision_index + layer_precision_count].reshape(
            out_channels, -1)

        sum_precision = reshaped_precision.sum(axis=1)

        num_filters_to_prune = int(prune_percentage * out_channels)

        _, indices_to_prune = torch.topk(torch.tensor(sum_precision), num_filters_to_prune, largest=False)

        for index in indices_to_prune:
            module.weight.data[index] = 0
            if module.bias is not None:
                module.bias.data[index] = 0
        num_pruned = len(indices_to_prune)

        precision_index += layer_precision_count
        num_bias_params = module.bias.numel() if module.bias is not None else 0
        if num_bias_params > 0:
            precision_index += num_bias_params

        return precision_index, num_pruned
    elif isinstance(module, (Bias, Scale)):
        
        precision_index += 1
    return precision_index, num_pruned