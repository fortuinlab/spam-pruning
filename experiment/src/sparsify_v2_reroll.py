import torch
import sys
import copy
import torch.nn as nn

from marglikopt import marglik_optimization
from laplace import KronLaplace, DiagLaplace 
import matplotlib.pyplot as plt
import numpy as np

from utils import check_sparsity, evaluate_classification
import wandb


def plot_histogram(H_facs_diag, bins=100):
    plt.hist(H_facs_diag, bins=bins, log=True)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of H_facs_diag values')
    plt.show()


def get_threshs(H_facs_diag, percentiles, plot=False):

    # This calculation is just for plotting purposes and np percentile could have been straight
    # used instead.
    q1, q2, q3, q4, q5,q6, q7, q8, q9, q10 = np.percentile(H_facs_diag, percentiles)
    

    # Group the values into quartiles
    q1_indices = np.where(H_facs_diag <= q1)[0]
    #q2_indices = np.where((H_facs_diag > q1) & (H_facs_diag <= q2))[0]
    q2_indices = np.where( (H_facs_diag <= q2))[0]
    #q3_indices = np.where((H_facs_diag > q2) & (H_facs_diag <= q3))[0]
    q3_indices = np.where((H_facs_diag <= q3))[0]
    #q4_indices = np.where(H_facs_diag > q3)[0]
    q4_indices = np.where(H_facs_diag < q4)[0]
    q5_indices = np.where(H_facs_diag < q5)[0]
    q6_indices = np.where(H_facs_diag < q6)[0]
    q7_indices = np.where(H_facs_diag < q7)[0]
    q8_indices = np.where(H_facs_diag < q8)[0]
    q9_indices = np.where(H_facs_diag < q9)[0]
    q10_indices = np.where(H_facs_diag < q10)[0]


    # Plot the bar graph with quartile annotations
    if plot:
        fig, ax = plt.subplots()
        ax.bar(q1_indices, H_facs_diag[q1_indices], color='blue', label='Q1')
        ax.bar(q2_indices, H_facs_diag[q2_indices], color='green', label='Q2')
        ax.bar(q3_indices, H_facs_diag[q3_indices], color='yellow', label='Q3')
        ax.bar(q4_indices, H_facs_diag[q4_indices], color='#FFAAAA', label='Q4')  # Light red color for Q4
        ax.bar(q5_indices, H_facs_diag[q5_indices], color='#FFAAAA', label='Q5')  
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')
        ax.set_yscale('log')  # Set the y-axis scale to logarithmic
        #ax.set_ylim(bottom=10**np.floor(np.log10(H_facs_diag.min())), top=10**np.ceil(np.log10(H_facs_diag.max())))  # Set the y-axis limits
        ax.legend(loc='upper right')  # Specify the legend location
        plt.show()
    return [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10]



def kron_laplacian_sparsity_strategy(la, model, test_loader, train_loader, sparsities,args):
    post_diag = la.posterior_precision.diag()
    #if H_facs_diag.is_cuda:
    #    H_facs_diag = H_facs_diag.cpu()
    wandb.log({"post_diag": post_diag})
    
    post_diag_np = post_diag.detach().cpu().numpy()


    weights = [param.view(-1) for param in model.parameters()]
    weights_sq = [w ** 2 for w in weights]
    weights_sq_flat = torch.cat(weights_sq)
    
    p_w = post_diag_np * weights_sq_flat.detach().cpu().numpy()
    wandb.log({"importance score": p_w, "weights":weights_sq_flat})
    thresh_values = get_threshs(p_w, sparsities, plot=False)
    models = {}
    for i in range(len(thresh_values)):
        model_copy = copy.deepcopy(model)
        with torch.no_grad():
            params = []
            for param in model_copy.parameters():
                params.append(param.view(-1))
            flat_params = torch.cat(params)

            mask = p_w < thresh_values[i]
            flat_params[mask] = 0.0

            index = 0
        for param in model_copy.parameters():
            numel = param.numel()
            new_param = flat_params[index:index + numel].view(param.shape)
            param.data = new_param.data
            index += numel

            # fine tune using marglik
        if args["fine_tune"]:   
            print("fine tuning")
            la, model_copy, margliks, perf = marglik_optimization(model = model_copy,
                                                            train_loader=train_loader,
                                                            valid_loader=test_loader,
                                                            laplace= KronLaplace,
                                                            #prior_structure="layerwise",
                                                            prior_structure=args["prior_structure"],
                                                            n_epochs=args['tune_epochs'],
                                                            n_epochs_burnin=args['tune_epochs_burnin'],
                                                            lr = args['lr'],
                                                            log_wandb = True)
            #apply the mask to the model again which enforces the sparsity
            with torch.no_grad():
                params = []
                for param in model_copy.parameters():
                    params.append(param.view(-1))
                flat_params = torch.cat(params)
                # We use the mask from before the fine tuning
                flat_params[mask] = 0.0

                index = 0
                for param in model_copy.parameters():
                    numel = param.numel()
                    new_param = flat_params[index:index + numel].view(param.shape)
                    param.data = new_param.data
                    index += numel


        val_acc = evaluate_classification(model_copy, test_loader)
        models[sparsities[i]] = {"model":model_copy, "val_acc": val_acc, "marglik": margliks if args["fine_tune"] else ["offline"]}
        print("sp", check_sparsity(model_copy))
        wandb.log({"sparsity": check_sparsity(model_copy), "val_acc": evaluate_classification(model_copy, test_loader),
                    "thresh": thresh_values[i], "target_sparsity": sparsities[i], "sparsity_type": "kron"})
    return models

    





def diag_laplacian_sparsity_strategy(la, model, test_loader, train_loader, sparsities, args):
    """

    Args:
        la (DiagLaplace): The DiagLaplace approximation.
    """
    post_diag = la.posterior_precision
    if post_diag.is_cuda:
        post_diag = post_diag.cpu()

    post_diag_np = post_diag.detach().numpy()
    #wandb.log({"post_diag": post_diag})
    weights = [param.view(-1) for param in model.parameters()]
    weights_sq = [w ** 2 for w in weights]
    weights_sq_flat = torch.cat(weights_sq)

    w_p = post_diag_np * weights_sq_flat.detach().cpu().numpy()
    #wandb.log({"importance score": w_p, "weights":weights_sq_flat})


    thresh_values = get_threshs(w_p, sparsities, plot=False)   
    models = {}
    for i in range(len(thresh_values)):
        model_copy = copy.deepcopy(model)
        with torch.no_grad():
            params = []
            for param in model_copy.parameters():
                params.append(param.view(-1))
            flat_params = torch.cat(params)

            mask = w_p < thresh_values[i]
            flat_params[mask] = 0.0

            index = 0
            for param in model_copy.parameters():
                numel = param.numel()
                new_param = flat_params[index:index + numel].view(param.shape)
                param.data = new_param.data
                index += numel

        if args["fine_tune"]:
            print("fine tuning")
            la, model_copy, margliks, perf = marglik_optimization(model = model_copy,
                                                            train_loader=train_loader,
                                                            valid_loader=test_loader,
                                                            laplace= DiagLaplace,
                                                            #laplace= KronLaplace,
                                                            prior_structure=args["prior_structure"],
                                                            #prior_structure="layerwise",	
                                                            n_epochs=args['tune_epochs'],
                                                            n_epochs_burnin=args['tune_epochs_burnin'],
                                                            lr=args['lr'],
                                                            log_wandb = True)
            #apply the mask to the model again which enforces the sparsity
            with torch.no_grad():
                params = []
                for param in model_copy.parameters():
                    params.append(param.view(-1))
                flat_params = torch.cat(params)

                flat_params[mask] = 0.0

                index = 0
                for param in model_copy.parameters():
                    numel = param.numel()
                    new_param = flat_params[index:index + numel].view(param.shape)
                    param.data = new_param.data
                    index += numel
                    
        val_acc = evaluate_classification(model_copy, test_loader)
        models[sparsities[i]] = {"model":model_copy, "val_acc": val_acc, "marglik": margliks if args["fine_tune"] else ["offline"]}
        print("sp", check_sparsity(model_copy))
        wandb.log({"sparsity": check_sparsity(model_copy), "val_acc": val_acc,
                    "thresh": thresh_values[i], "target_sparsity": sparsities[i], "sparsity_type": "diag"})
    return models

def laplace_abs(la, model,test_loader, train_loader, sparsities,args):
    post_diag = la.posterior_precision.diag()
    post_diag_np = post_diag.detach().cpu().numpy()
    post_diag_abs = np.abs(post_diag_np)


    weights = [param.view(-1) for param in model.parameters()]
    weights_sq = [w ** 2 for w in weights]
    weights_sq_flat = torch.cat(weights_sq)
    
    p_w = post_diag_abs * weights_sq_flat.detach().cpu().numpy()

    thresh_values = get_threshs(p_w, sparsities, plot=False)

    models = {}
    for i in range(len(thresh_values)):
        model_copy = copy.deepcopy(model)
        with torch.no_grad():
            params = []
            for param in model_copy.parameters():
                params.append(param.view(-1))
            flat_params = torch.cat(params)

            mask = p_w < thresh_values[i]
            flat_params[mask] = 0.0

            index = 0
            for param in model_copy.parameters():
                numel = param.numel()
                new_param = flat_params[index:index + numel].view(param.shape)
                param.data = new_param.data
                index += numel

        # fine tune using marglik
        #la, model_copy, margliks, perf = marglik_optimization(model = model_copy,
        #                                                train_loader=train_loader,
        #                                                valid_loader=test_loder,
        #                                                n_epochs=1,
        #                                                lr = args['lr'],
        #                                                log_wandb = True)


        models[i] = model_copy
        
        print("sp", check_sparsity(model_copy))
        wandb.log({"sparsity": check_sparsity(model_copy), "val_acc": evaluate_classification(model_copy, test_loader),
                    "thresh": thresh_values[i], "target_sparsity": sparsities[i], "sparsity_type": "kron"})
    return models






def sparse_strategy(la, model,test_loader, train_loader, sparsities,args):
    """
    Returns a sparsity strategy for the given Laplace approximation.

    Args:
        la (LaplaceApproximation): The Laplace approximation.
        model (torch.nn.Module): The model to sparsify.

    Returns:
        A sparsity strategy.
    """
    if isinstance(la, KronLaplace):
        return kron_laplacian_sparsity_strategy(la, model,test_loader, train_loader, sparsities,args)
    elif isinstance(la, DiagLaplace):
        return diag_laplacian_sparsity_strategy(la, model,test_loader, train_loader, sparsities,args)
    else:
        raise ValueError
    


def SNIP_strategy(la, model, test_loader, train_loader, sparsities, args):
    """ Single-shot network pruning (SNIP) strategy. """
    criterion = nn.CrossEntropyLoss()
    models = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    laplace = KronLaplace if isinstance(la, KronLaplace) else DiagLaplace

    for sparsity in sparsities:
        model_copy = copy.deepcopy(model).to(device)
        flat_params = torch.nn.utils.parameters_to_vector(model_copy.parameters())

        # Compute gradients for the parameters
        grads = [torch.zeros_like(param, device=device) for param in model_copy.parameters()]
        for param in model_copy.parameters():
            param.requires_grad_(True)

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model_copy(data)
            loss = criterion(output, target)
            model_copy.zero_grad()
            loss.backward()

            for i, param in enumerate(model_copy.parameters()):
                grads[i] += param.grad.abs() / len(train_loader)

        # Compute the mask
        importance_scores = torch.cat([grad.view(-1) for grad in grads])
        _, idxs = torch.sort(importance_scores)
        flat_params[idxs[:int(len(idxs) * sparsity / 100)]] = 0
        torch.nn.utils.vector_to_parameters(flat_params, model_copy.parameters())

        # Fine-tuning
        if args["fine_tune"]:
            print("fine tuning")
            la, model_copy, margliks, perf = marglik_optimization(model=model_copy,
                                                            train_loader=train_loader,
                                                            valid_loader=test_loader,
                                                            laplace=laplace,
                                                            prior_structure=args["prior_structure"],
                                                            n_epochs=args['tune_epochs'],
                                                            n_epochs_burnin=args['tune_epochs_burnin'],
                                                            lr = args['lr'],
                                                            log_wandb=True)

            # Reapply the mask to ensure sparsity
            with torch.no_grad():
                flat_params = torch.nn.utils.parameters_to_vector(model_copy.parameters())
                flat_params[idxs[:int(len(idxs) * sparsity / 100)]] = 0
                torch.nn.utils.vector_to_parameters(flat_params, model_copy.parameters())

        val_acc = evaluate_classification(model_copy, test_loader)
        models[sparsity] = {"model":model_copy, "val_acc": val_acc, "marglik": margliks if args["fine_tune"] else ["offline"]}
        print("sp", check_sparsity(model_copy))
        wandb.log({"sparsity": check_sparsity(model_copy), "val_acc": val_acc,
                   "target_sparsity": sparsity, "sparsity_type": "SNIP"})

    return models



def GraSP_strategy(la, model, test_loader, train_loader, sparsities,args):
    """ Gradient Signal Preservation"""
    criterion = nn.CrossEntropyLoss() 
    models = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    laplace = KronLaplace if isinstance(la, KronLaplace) else DiagLaplace
    
    for sparsity in sparsities:
        model_copy = copy.deepcopy(model)
        total_params = sum([np.prod(p.shape) for p in model_copy.parameters()])
        num_params_to_prune = int(total_params * (sparsity/100))

        # Flatten all the parameters and move them to the CPU
        params = []
        for param in model_copy.parameters():
            params.append(param.cpu().view(-1))
        flat_params = torch.cat(params)

        # Compute gradients for the parameters and the importance scores for the parameters which is the product of the gradient and the parameter
        grads = []
        for param in model_copy.parameters():
            param.requires_grad_(True)
        grads = []

        for param in model_copy.parameters():
            grads.append(torch.zeros_like(param))
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            output = model_copy(data)
            loss = criterion(output, target)
            loss.backward()
            for i, param in enumerate(model_copy.parameters()):
                grads[i] += param.grad.abs() / len(train_loader) 
        
        # importance score = gradient * parameter

        importance_scores = []  
        for param, grad in zip(model_copy.parameters(), grads):
            importance_scores.append(torch.abs(grad) * torch.abs(param))

        # Compute the mask
        importance_scores = [importance_score.view(-1) for importance_score in importance_scores]
        importance_scores = torch.cat(importance_scores)
        _, idxs = torch.sort(importance_scores)
        flat_params[idxs[:num_params_to_prune]] = 0.0

        # Put the flattened parameters back in the model
        index = 0
        for param in model_copy.parameters():
            numel = param.numel()
            new_param = flat_params[index:index + numel].view(param.shape)
            param.data = new_param.data
            index += numel

        # fine tune using marglik
        model_copy.to("cpu")
        if args["fine_tune"]:
            print("fine tuning")
            la, model_copy, margliks, perf = marglik_optimization(model = model_copy,
                                                            train_loader=train_loader,
                                                            valid_loader=test_loader,
                                                            laplace= laplace,
                                                            prior_structure=args["prior_structure"],
                                                            n_epochs=args['tune_epochs'],
                                                            n_epochs_burnin=args['tune_epochs_burnin'],
                                                            lr = args['lr'],
                                                            log_wandb = True)
            # apply the mask to the model again which enforces the sparsity
            with torch.no_grad():
                params = []
                for param in model_copy.parameters():
                    params.append(param.view(-1))
                flat_params = torch.cat(params)

                flat_params[idxs[:num_params_to_prune]] = 0.0

                index = 0
                for param in model_copy.parameters():
                    numel = param.numel()
                    new_param = flat_params[index:index + numel].view(param.shape)
                    param.data = new_param.data
                    index += numel
        
        val_acc = evaluate_classification(model_copy, test_loader)
        models[sparsity] = {"model":model_copy, "val_acc": val_acc, "marglik": margliks if args["fine_tune"] else ["offline"]}
        print("sp", check_sparsity(model_copy))
        wandb.log({"sparsity": check_sparsity(model_copy), "val_acc":val_acc,
                    "target_sparsity": sparsity, "sparsity_type": "GraSP"})
    return models
    

                


def magnitude_pruning(la, model, test_loader, train_loader, sparsities, args):
    """
    Implements magnitude pruning strategy.

    Args:
        model (torch.nn.Module): The model to prune.
        test_loader (DataLoader): The validation/test data loader.
        sparsities (list): A list of target sparsity levels.

    Returns:
        A dictionary of pruned models with corresponding target sparsity levels.
    """
    models = {}
    laplace = KronLaplace if isinstance(la, KronLaplace) else DiagLaplace

    for sparsity in sparsities:
        model_copy = copy.deepcopy(model)

        with torch.no_grad():
            flat_params = torch.nn.utils.parameters_to_vector(model_copy.parameters())
            
            threshold = flat_params.abs().quantile(sparsity / 100.0)
            
            mask = torch.abs(flat_params) < threshold
            flat_params[mask] = 0

            torch.nn.utils.vector_to_parameters(flat_params, model_copy.parameters())

        if args["fine_tune"]:
            print("fine tuning")
            la, model_copy, margliks, perf = marglik_optimization(model=model_copy,
                                                            train_loader=train_loader,
                                                            valid_loader=test_loader,
                                                            laplace=laplace,
                                                            prior_structure=args["prior_structure"],
                                                            n_epochs=args['tune_epochs'],
                                                            n_epochs_burnin=args['tune_epochs_burnin'],
                                                            lr = args['lr'],
                                                            log_wandb=True)
            
            # Reapply the mask to enforce sparsity
            with torch.no_grad():
                flat_params = torch.nn.utils.parameters_to_vector(model_copy.parameters())
                flat_params[mask] = 0
                torch.nn.utils.vector_to_parameters(flat_params, model_copy.parameters())

        val_acc = evaluate_classification(model_copy, test_loader)
        models[sparsity] = {"model":model_copy, "val_acc": val_acc, "marglik": margliks if args["fine_tune"] else ["offline"]}
        print(f"Sparsity: {check_sparsity(model_copy)}, Accuracy: {val_acc}")
        wandb.log({"sparsity": check_sparsity(model_copy), "val_acc": val_acc, "target_sparsity": sparsity})

    return models




def random_strategy(la, model, test_loader, train_loader, sparsities, args):
    """
    Implements random pruning strategy.

    Args:
        model (torch.nn.Module): The model to prune.
        sparsities (list): A list of target sparsity levels.

    Returns:
        A dictionary of pruned models with corresponding target sparsity levels.
    """
    models = {}
    laplace = KronLaplace if isinstance(la, KronLaplace) else DiagLaplace
    
    for sparsity in sparsities:
        model_copy = copy.deepcopy(model)
        
        with torch.no_grad():
            flat_params = torch.nn.utils.parameters_to_vector(model_copy.parameters())
            
            # Calculate number of parameters to prune
            num_params_to_prune = int(flat_params.numel() * (sparsity/100))
            
            # Randomly select parameters and set them to zero
            indices_to_prune = torch.randperm(flat_params.numel())[:num_params_to_prune]
            flat_params[indices_to_prune] = 0.0
            
            torch.nn.utils.vector_to_parameters(flat_params, model_copy.parameters())

        # Fine-tuning
        if args["fine_tune"]:
            print("fine tuning")
            la, model_copy, margliks, perf = marglik_optimization(model=model_copy,
                                                            train_loader=train_loader,
                                                            valid_loader=test_loader,
                                                            laplace=laplace,
                                                            prior_structure=args["prior_structure"],
                                                            n_epochs=args['tune_epochs'],
                                                            n_epochs_burnin=args['tune_epochs_burnin'],
                                                            lr = args['lr'],
                                                            log_wandb=True)
            
            # Reapply the mask to enforce sparsity
            with torch.no_grad():
                flat_params = torch.nn.utils.parameters_to_vector(model_copy.parameters())
                flat_params[indices_to_prune] = 0.0
                torch.nn.utils.vector_to_parameters(flat_params, model_copy.parameters())

        val_acc = evaluate_classification(model_copy, test_loader)
        models[sparsity] = {"model":model_copy, "val_acc": val_acc, "marglik": margliks if args["fine_tune"] else ["offline"]}
        print("sparsity:", check_sparsity(model_copy))
        
        wandb.log({"sparsity": check_sparsity(model_copy), "val_acc": val_acc,
                "target_sparsity": sparsity})

    return models


def obd(la, model, test_loader, train_loader, sparsities, args):
    """
    Implements optimal brain damage strategy.
    
    Args:
        model (torch.nn.Module): The model to prune.
        test_loader (DataLoader): The validation/test data loader.
        sparsities (list): A list of target sparsity levels.
        
    Returns:
        A dictionary of pruned models with corresponding target sparsity levels.
    """
    
    