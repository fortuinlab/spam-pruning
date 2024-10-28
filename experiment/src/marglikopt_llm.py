from copy import deepcopy
import math
import numpy as np
import torch
from torch.nn.utils.convert_parameters import vector_to_parameters
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.utils import parameters_to_vector
from torch.distributions import Normal
import torch.nn as nn
from collections import UserDict
import wandb



from laplace.utils import  UnitPrior   # ,expand_prior_precision, fix_prior_prec_structure #using these functions instead of the ones below
from laplace import KronLaplace
from laplace.curvature import AsdlGGN, AsdlEF

from structured_sparsity import apply_masks
from tqdm import tqdm

GB_FACTOR = 1024 ** 3

#TODO ASK if we care only about trainable parameters or all parameters

def get_num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_sparsity_all(model):
    num_zeros = 0
    num_params = 0
    for _, param in model.named_parameters():
        num_zeros += torch.sum(param == 0).item()
        num_params += param.numel()
    return num_zeros / num_params

def get_sparsity(model):
    # trainable only parameters
    num_zeros = 0
    num_params = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            num_zeros += torch.sum(param == 0).item()
            num_params += param.numel()
    #print("num_zeros: ", num_zeros)
    #print("num_params: ", num_params)
    return num_zeros / num_params


def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def none_zero_params(model):
    none_zero = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            none_zero += torch.sum(module.weight != 0.0)
    return none_zero


def get_zero_ratio(model):
    return 1-(none_zero_params(model) / num_params(model))






"""def expand_prior_precision(prior_prec, model):
    theta = parameters_to_vector(model.parameters())
    device, P = theta.device, len(theta)
    assert prior_prec.ndim == 1
    if len(prior_prec) == 1:  # scalar
        return torch.ones(P, device=device) * prior_prec
    elif len(prior_prec) == P:  # full diagonal
        return prior_prec.to(device)
    else:
        return torch.cat([delta * torch.ones_like(m).flatten() for delta, m
                          in zip(prior_prec, model.parameters())])"""


"""def get_prior_hyperparams(prior_prec_init, prior_structure, H, P, device, model=None):
    log_prior_prec_init = np.log(prior_prec_init)
    if prior_structure == 'scalar':
        log_prior_prec = log_prior_prec_init * torch.ones(1, device=device)
    elif prior_structure == 'layerwise':
        log_prior_prec = log_prior_prec_init * torch.ones(H, device=device)
    elif prior_structure == 'diagonal':
        log_prior_prec = log_prior_prec_init * torch.ones(P, device=device)
    elif prior_structure == 'unitwise':
        assert model is not None, 'unitwise prior requires specifying the model'
        prior_prec_init = UnitPrior.init_from_model(model) * prior_prec_init
    else:
        raise ValueError(f'Invalid prior structure {prior_structure}')
    log_prior_prec.requires_grad = True
    return log_prior_prec"""

# having the functions here for quick  access/changes
def expand_prior_precision(prior_prec, model):
    """Expand prior precision to match the shape of the model parameters.

    Parameters
    ----------
    prior_prec : torch.Tensor 1-dimensional
        prior precision
    model : torch.nn.Module
        torch model with parameters that are regularized by prior_prec

    Returns
    -------
    expanded_prior_prec : torch.Tensor
        expanded prior precision has the same shape as model parameters
    """
    theta = parameters_to_vector(model.parameters())
    device, P = theta.device, len(theta)
    if isinstance(prior_prec, UnitPrior):
        return prior_prec.diag()
    assert prior_prec.ndim == 1
    if len(prior_prec) == 1:  # scalar
        return torch.ones(P, device=device) * prior_prec
    elif len(prior_prec) == P:  # full diagonal
        return prior_prec.to(device)
    else:
        return torch.cat([delta * torch.ones_like(m).flatten() for delta, m
                          in zip(prior_prec, model.parameters())])


def fix_prior_prec_structure(prior_prec_init, prior_structure, n_layers, n_params, device, dtype=None, model=None):
    if prior_structure == 'scalar':
        prior_prec_init = torch.full((1,), prior_prec_init, device=device, dtype=dtype)   
    elif prior_structure == 'layerwise':
        prior_prec_init = torch.full((n_layers,), prior_prec_init, device=device, dtype=dtype)
    elif prior_structure == 'diagonal':
        prior_prec_init = torch.full((n_params,), prior_prec_init, device=device, dtype=dtype)
    elif prior_structure == 'unitwise':
        assert model is not None, 'unitwise prior requires specifying the model'
        prior_prec_init = UnitPrior.init_from_model(model) * prior_prec_init
    else:
        raise ValueError(f'Invalid prior structure {prior_structure}.')
    return prior_prec_init




def valid_performance(model, test_loader, likelihood, criterion, device):
    N = len(test_loader.dataset)
    perf = 0
    nll = 0
    if len(test_loader.dataset[0]) == 2:
        for X, y in test_loader:
            X, y = X.detach().to(device), y.detach().to(device)
            with torch.no_grad():
                f = model(X)
            if likelihood == 'classification':
                perf += (torch.argmax(f, dim=-1) == y).sum() / N
            elif likelihood == 'language_modeling':
                perf += math.exp(X.size(0) * criterion(f,y).item() / N)
            elif likelihood == 'heteroscedastic_regression':
                perf += (y.squeeze() + 0.5 * f[:, 0] / f[:, 1]).square().sum() / N
            else:
                perf += (f - y).square().sum() / N
            nll += criterion(f, y) / len(test_loader)
        
    else:
        for batch in test_loader:
            labels = batch['labels'].to(device)
            with torch.no_grad():
                f = model(batch)
            if likelihood == 'classification':
                perf+= (torch.argmax(f, dim=-1) == labels).sum() / N
            nll += criterion(f, labels) / len(test_loader)
    return perf.item(), nll.item()
                


def get_scheduler(scheduler, optimizer, train_loader, n_epochs, lr, lr_min):
    n_steps = n_epochs * len(train_loader)
    if scheduler == 'exp':
        min_lr_factor = lr_min / lr
        gamma = np.exp(np.log(min_lr_factor) / n_steps)
        return ExponentialLR(optimizer, gamma=gamma)
    elif scheduler == 'cos':
        return CosineAnnealingLR(optimizer, n_steps, eta_min=lr_min)
    else:
        raise ValueError(f'Invalid scheduler {scheduler}')


def get_model_optimizer(optimizer, model, lr, weight_decay=0):
    if optimizer == 'adam':
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == 'adamw':
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == 'sgd':
        # fixup parameters should have 10x smaller learning rate
        is_fixup = lambda param: param.size() == torch.Size([1])  # scalars
        fixup_params = [p for p in model.parameters() if is_fixup(p)]
        standard_params = [p for p in model.parameters() if not is_fixup(p)]
        params = [{'params': standard_params}, {'params': fixup_params, 'lr': lr / 10.}]
        return SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer == 'sgd_standard':
        return SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f'Invalid optimizer {optimizer}')


def gradient_to_vector(parameters):
    return parameters_to_vector([e.grad for e in parameters])



def vector_to_gradient(vec, parameters):
    return vector_to_parameters(vec, [e.grad for e in parameters])



def marglik_optimization(model,
                         train_loader,
                         valid_loader=None,
                         likelihood='classification',
                         prior_structure='layerwise',
                         #prior_prec_init=1.,
                         prior_prec_init=1.,
                         sigma_noise_init=1.,
                         temperature=1.,
                         n_epochs=500,
                         lr=1e-3,
                         lr_min=None,
                         optimizer='adam',
                         scheduler='cos',
                         n_epochs_burnin=0,
                         n_hypersteps=100,
                         marglik_frequency=1,
                         lr_hyp=1e-1,
                         lr_hyp_min=1e-1,
                         laplace=KronLaplace,
                         backend=AsdlEF,
                         early_stopping=False,
                         log_wandb = False,
                         mask_dict = None,
                         ):
    """Runs marglik optimization training for a given model and training dataloader.

    Parameters
    ----------
    model : torch.nn.Module
        torch model
    train_loader : DataLoader
        pytorch training dataset loader
    valid_loader : DataLoader
    likelihood : str
        'classification', 'regression', 'heteroscedastic_regression'
    prior_structure : str
        'scalar', 'layerwise', 'diagonal'
    prior_prec_init : float
        initial prior precision
    sigma_noise_init : float
        initial observation noise (for regression only)
    temperature : float
        factor for the likelihood for 'overcounting' data.
        Often required when using data augmentation.
    n_epochs : int
    lr : float
        learning rate for model optimizer
    lr_min : float
        minimum learning rate, defaults to lr and hence no decay
        to have the learning rate decay from 1e-3 to 1e-6, set
        lr=1e-3 and lr_min=1e-6.
    optimizer : str
        either 'adam' or 'sgd'
    scheduler : str
        either 'exp' for exponential and 'cos' for cosine decay towards lr_min
    n_epochs_burnin : int default=0
        how many epochs to train without estimating and differentiating marglik
    n_hypersteps : int
        how many steps to take on the hyperparameters when marglik is estimated
    marglik_frequency : int
        how often to estimate (and differentiate) the marginal likelihood
    lr_hyp : float
        learning rate for hyperparameters (should be between 1e-3 and 1)
    laplace : Laplace
        type of Laplace approximation (Kron/Diag/Full)
    backend : Backend
        AsdlGGN/AsdlEF or BackPackGGN/BackPackEF
    stochastic_grad : bool
    independent : bool
        whether to use independent functional laplace
    single_output : bool
        whether to use single random output for functional laplace
    kron_jac : bool
        whether to use kron_jac in the backend

    Returns
    -------
    lap : Laplace
        lapalce approximation
    model : torch.nn.Module
    margliks : list
    losses : list
    """
    # for transformer model only 
    sparsity_check = True
    if sparsity_check:
        # check if model is sparse
        initial_sparsity = get_zero_ratio(model)
        print(f'Initial sparsity: {initial_sparsity:.4f}')
    

    if lr_min is None:  # don't decay lr
        lr_min = lr
    device = parameters_to_vector(model.parameters()).device
    dtype = parameters_to_vector(model.parameters()).dtype
    N = len(train_loader.dataset)
    H = len(list(model.parameters()))
    P = len(parameters_to_vector(model.parameters()))
    best_model_dict = None

    # differentiable hyperparameters
    hyperparameters = list()
    # prior precision
    log_prior_prec_init = np.log(temperature * prior_prec_init)
    log_prior_prec = fix_prior_prec_structure(
        log_prior_prec_init, prior_structure, H, P, device, dtype=dtype, model=model)
    log_prior_prec.requires_grad = True
    if isinstance(log_prior_prec, UnitPrior):
        log_prior_prec.log = True
        hyperparameters.extend(log_prior_prec.parameters())
    else:
        hyperparameters.append(log_prior_prec)

    # set up loss (and observation noise hyperparam)
    if likelihood == 'classification' or likelihood == 'language_modelling':
        criterion = CrossEntropyLoss(reduction='mean')
        sigma_noise = 1
    elif likelihood == 'regression':
        criterion = MSELoss(reduction='mean')
        log_sigma_noise_init = np.log(sigma_noise_init)
        log_sigma_noise = log_sigma_noise_init * torch.ones(1, device=device)
        log_sigma_noise.requires_grad = True
        hyperparameters.append(log_sigma_noise)
    else:
        raise ValueError()

    # set up model optimizer and scheduler
    optimizer = get_model_optimizer(optimizer, model, lr)
    scheduler = get_scheduler(scheduler, optimizer, train_loader, n_epochs, lr, lr_min)

    n_steps = ((n_epochs - n_epochs_burnin) // marglik_frequency) * n_hypersteps
    hyper_optimizer = Adam(hyperparameters, lr=lr_hyp)
    hyper_scheduler = CosineAnnealingLR(hyper_optimizer, n_steps, eta_min=lr_hyp_min)

    losses = list()
    valid_perfs = list()
    valid_nlls = list()
    margliks = list()
    best_marglik = np.inf

    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0
        epoch_perf = 0
        epoch_nll = 0
        epoch_log = dict(epoch=epoch)

        # standard NN training per batch
        torch.cuda.empty_cache()
        #for X, y in train_loader:
        #    X, y = X.detach().to(device), y.to(device)
        for data in tqdm(train_loader):
            if isinstance(data, UserDict) or isinstance(data, dict):
                X, y = data, data['labels']
                y = y.to(device, non_blocking=True)
            else:
                X, y = data
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            # rest of the code
            if isinstance(data, UserDict) or isinstance(data, dict):
                X, y = data, data['labels']
                y = y.to(device, non_blocking=True)
            else:
                X, y = data
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            #print(f'X shape: {X.shape}')
            #print(f'y shape: {y.shape}')
            optimizer.zero_grad()

            if likelihood == 'regression':
                sigma_noise = torch.exp(log_sigma_noise).detach()
                crit_factor = 1 / temperature / (2 * sigma_noise.square())
            else:
                crit_factor = prior_prec_init / temperature
            if isinstance(log_prior_prec, UnitPrior):
                delta = torch.exp(log_prior_prec.diag()).detach()
            else:
                prior_prec = torch.exp(log_prior_prec).detach()
                delta = expand_prior_precision(prior_prec, model)
            
            f = model(X)
            if likelihood == 'language_modelling':
                loss = f.loss
            else:
                loss = criterion(f, y)
            theta = parameters_to_vector(model.parameters())
            loss +=  (0.5 * (delta * theta) @ theta) / N / crit_factor
            #print(f'loss: {loss:.5f}')
            loss.backward()
            optimizer.step()
            # Applying mask after optim
            if mask_dict is not None:
                apply_masks(model, mask_dict)
                
            epoch_loss += loss.cpu().item() / len(train_loader)
            if likelihood == 'language_modelling':
                epoch_nll += f.loss.item() / len(train_loader)
            else:
                epoch_nll += criterion(f.detach(), y).item() / len(train_loader)
                
            if likelihood == 'regression':
                epoch_perf += (f.detach() - y).square().sum() / N
            elif likelihood == 'heteroscedastic_regression':
                epoch_perf += (y.squeeze() + 0.5 * f[:, 0] / f[:, 1]).square().sum() / N
            elif likelihood == 'language_modelling':
                epoch_perf += math.exp(X.input_ids.size(0) * f.loss.item()) / N
            else:
                epoch_perf += torch.sum(torch.argmax(f.detach(), dim=-1) == y).item() / N
            scheduler.step()

        losses.append(epoch_loss)
        #logging.info(f'MARGLIK[epoch={epoch}]: train. perf={epoch_perf:.2f}; loss={epoch_loss:.5f}; nll={epoch_nll:.5f}')
        optimizer.zero_grad(set_to_none=True)
        llr = scheduler.get_last_lr()[0]
        epoch_log.update({'train/loss': epoch_loss, 'train/nll': epoch_nll, 'train/perf': epoch_perf, 'train/lr': llr})
        if log_wandb == True:
            wandb.log(epoch_log)
        # compute validation error to report during training
        if valid_loader is not None:
            with torch.no_grad():
                if likelihood == 'regression':
                    def val_criterion(f, y):
                        assert f.shape == y.shape
                        log_lik = Normal(loc=f, scale=sigma_noise).log_prob(y)
                        return -log_lik.mean()
                else:
                    val_criterion = criterion
                val_perf, val_nll = valid_performance(model, valid_loader, likelihood, val_criterion, device)
                valid_perfs.append(val_perf)
                valid_nlls.append(val_nll)
                #logging.info(f'MARGLIK[epoch={epoch}]: valid. perf={val_perf:.2f}; nll={val_nll:.5f}.')
                epoch_log.update({'valid/perf': val_perf, 'valid/nll': val_nll})
                if log_wandb == True:
                    wandb.log(epoch_log)
                #if epoch > 3 and val_perf < 0.15:
                #    print("Validation accuracy is below 0.1. Stopping training.")
                #    wandb.log({'message': 'Validation accuracy is below 0.1. Stopping training.'})
                #    break
                if epoch > 3 and val_perf < 0.1:
                    print("Validation accuracy is below 0.1. Stopping training.")
                    wandb.log({'message': 'Validation accuracy is below 0.1. Stopping training.'})
                    break
                #wandb.log(epoch_log)
        # only update hyperparameters every "Frequency" steps after "burnin"
        if (epoch % marglik_frequency) != 0 or epoch < n_epochs_burnin:
            continue

        # 1. fit laplace approximation
        torch.cuda.empty_cache()

        sigma_noise = 1 if likelihood != 'regression' else torch.exp(log_sigma_noise)
        
        if isinstance(log_prior_prec, UnitPrior):
            prior_prec = torch.exp(log_prior_prec.diag())
        else:
            prior_prec = torch.exp(log_prior_prec)
            
        lap = laplace(model, likelihood, sigma_noise=sigma_noise, prior_precision=prior_prec,
                        temperature=temperature, backend=backend)
        lap.fit(train_loader)
        # first optimize prior precision jointly with direct marglik grad
        margliks_local = list()
        for i in range(n_hypersteps):
            hyper_optimizer.zero_grad()
            sigma_noise = None if likelihood != 'regression' else torch.exp(log_sigma_noise)
            if isinstance(log_prior_prec, UnitPrior):
                prior_prec = torch.exp(log_prior_prec.diag())
            else:
                prior_prec = torch.exp(log_prior_prec)
            marglik = -lap.log_marginal_likelihood(prior_prec, sigma_noise) / N
            marglik.backward()
            margliks_local.append(marglik.item())
            hyper_optimizer.step()
            hyper_scheduler.step()

        marglik = margliks_local[-1]

        if likelihood == 'regression':
            epoch_log['hyperparams/sigma_noise'] = torch.exp(log_sigma_noise.detach()).cpu().item()
        epoch_log['train/marglik'] = marglik
        if log_wandb == True:
            wandb.log(epoch_log)
        margliks.append(marglik)
        del lap

        # sparsity check 
        if sparsity_check == True:
            sp = get_zero_ratio(model)
            if log_wandb == True:
                wandb.log({'sparsity_ep': sp})



        # early stopping on marginal likelihood
        if early_stopping and (margliks[-1] < best_marglik):
            best_model_dict = deepcopy(model.state_dict())
            if isinstance(log_prior_prec, UnitPrior):
                best_precision = deepcopy(torch.exp(log_prior_prec.diag().detach()))
            else:
                best_precision = deepcopy(prior_prec.detach())
            best_sigma = 1 if likelihood != 'regression' else deepcopy(sigma_noise.detach())
            best_marglik = margliks[-1]

    if early_stopping and (best_model_dict is not None):
        model.load_state_dict(best_model_dict)
        sigma_noise = best_sigma
        prior_prec = best_precision
    else:
        sigma_noise = 1 if sigma_noise is None else sigma_noise

    lap = laplace(model, likelihood, sigma_noise=sigma_noise, prior_precision=prior_prec,
                  temperature=temperature, backend=backend)
    lap.fit(train_loader)

    # additional just to know the value of the not preset hyperparameters passed from the config in case there will be a corrolation

    marglik_param = {
    'likelihood': likelihood,
    'prior_structure': prior_structure,
    'prior_prec_init': prior_prec_init,
    'sigma_noise_init': sigma_noise_init,
    'temperature': temperature,
    'n_epochs': n_epochs,
    'lr': lr,
    'lr_min': lr_min,
    'optimizer': optimizer,
    'scheduler': scheduler,
    'n_epochs_burnin': n_epochs_burnin,
    'n_hypersteps': n_hypersteps,
    'marglik_frequency': marglik_frequency,
    'lr_hyp': lr_hyp,
    'lr_hyp_min': lr_hyp_min,
    'laplace': "laplace",
    'backend': backend,
    'early_stopping': early_stopping
    }
    if log_wandb == True:
        wandb.config.update({"marglik_param":marglik_param}, allow_val_change=True)
    
    # returned model sparsity 
    sp_final = get_zero_ratio(model)
    epoch_log.update({'final_sparsity': sp_final})

    if log_wandb == True:
        wandb.log(epoch_log)

    if valid_loader is None:
        return lap, model, margliks
    else:
        return lap, model, margliks, val_perf