import torch


def synflow_score(model, dataloader, device, criterion):
    # Linearize the model
    signs = {}
    for name, param in model.named_parameters():
        signs[name] = torch.sign(param.data)
        param.data.abs_()

    # Forward pass with a dummy input
    input_dim = list(next(iter(dataloader))[0].shape)[1:]
    input = torch.ones([1] + input_dim).to(device)
    output = model(input)
    torch.sum(output).backward()

    # Compute scores
    scores = {}
    for name, param in model.named_parameters():
        scores[name] = torch.clone(param.grad * param).detach().abs_()
        param.grad.data.zero_()

    # Nonlinearize the model
    for name, param in model.named_parameters():
        param.data.mul_(signs[name])
    
    return scores




def snip_score(model, dataloader, device, criterion):
    # Enable gradients for masks
    for param in model.parameters():
        if param.requires_grad:
            param.requires_grad = True

    # Compute gradients
    scores = {}
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                scores[name] = torch.clone(param.grad).detach().abs_()
                param.grad.data.zero_()

    return scores

def grasp_score(model, dataloader, device, criterion):
    # Initialize stopped_grads as a zero tensor
    stopped_grads = None

    # First gradient pass
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)
        
        # Initialize stopped_grads if not done yet
        if stopped_grads is None:
            stopped_grads = torch.cat([g.reshape(-1) for g in grads]).detach()
        else:
            stopped_grads += torch.cat([g.reshape(-1) for g in grads])

    # Second gradient pass
    scores = {}
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        gnorm = (stopped_grads * torch.cat([g.reshape(-1) for g in grads])).sum()
        gnorm.backward()

        # Calculate and store the absolute scores per named parameter
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Use absolute value of the product of gradient and parameter
                scores[name] = torch.abs(param.grad * param.data).detach()
                param.grad.data.zero_()

    return scores

def magnitude_score(model):
    scores = {}
    for name, param in model.named_parameters():
        scores[name] = torch.clone(param.data).detach().abs_()
    
    return scores

def random_score(model, dataloader, device, criterion):
    scores = {}
    for name, param in model.named_parameters():
        scores[name] = torch.rand_like(param.data)
    
    return scores


    
