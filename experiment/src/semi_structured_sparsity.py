import torch
import torch.nn.functional as F

def apply_2_4_sparsity_with_padding(weights_flat, importance_scores, permute=False):
    original_length = len(weights_flat)
    padding = (4 - original_length % 4) % 4


    weights_padded = F.pad(weights_flat, (0, padding), 'constant', 0)
    importance_padded = F.pad(importance_scores, (0, padding), 'constant', 0)

    if permute:
        sorted_indices = torch.argsort(importance_padded, descending=True)
        weights_padded = weights_padded[sorted_indices]


    weights_grouped = weights_padded.view(-1, 4)
    for i in range(weights_grouped.size(0)):
        _, indices_to_zero = torch.topk(weights_grouped[i], 2, largest=False)
        weights_grouped[i][indices_to_zero] = 0

    if permute:
        # Flatten to reverse the permutation
        weights_flat = weights_grouped.view(-1)
        inverse_indices = torch.argsort(sorted_indices)
        weights_flat = weights_flat[inverse_indices]

 
    return weights_flat[:original_length]


def apply_2_4_sparsity_magnitude_with_padding(weights_flat, permute=False):
    original_length = len(weights_flat)
    padding = (4 - original_length % 4) % 4

    weights_padded = F.pad(weights_flat, (0, padding), 'constant', 0)

    sorted_indices = None
    if permute:
        sorted_indices = torch.argsort(torch.abs(weights_padded), descending=True)
        weights_padded = weights_padded[sorted_indices]

    weights_grouped = weights_padded.view(-1, 4)

    for i in range(weights_grouped.size(0)):
        _, indices_to_zero = torch.topk(torch.abs(weights_grouped[i]), 2, largest=False)
        weights_grouped[i][indices_to_zero] = 0

    if permute and sorted_indices is not None:
        weights_flat = weights_grouped.view(-1)
        inverse_indices = torch.argsort(sorted_indices).to(weights_flat.device)
        weights_flat = weights_flat[inverse_indices]
        return weights_flat[:original_length]
    else:
        return weights_grouped.view(-1)[:original_length]
    
    
    