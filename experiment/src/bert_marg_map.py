import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertConfig, DistilBertForSequenceClassification
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from marglikopt import marglik_optimization
# Set random seeds for reproducibility
import numpy as np
import wandb
import random
from laplace import DiagLaplace
from collections.abc import MutableMapping
# Logging setup
logging.basicConfig(level=logging.INFO)

seed = np.random.randint(0, 1000)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
import copy
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from sparsify_v2 import (sparse_strategy, magnitude_pruning, random_strategy, 
                         SNIP_strategy, GraSP_strategy)


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Load and prepare dataset
dataset = load_dataset("imdb")
train_dataset = dataset['train'].shuffle(seed=42)
test_dataset = dataset['test'].shuffle(seed=42)

# Print a few examples
print(train_dataset[0])

# Tokenization

def config_Wb(config):
    unique_id = wandb.util.generate_id()
    run_name = f"Distilbert_IMDB_{config['num_epochs']}"
    if config['laplace']['num_epochs_burnin'] > config['num_epochs']:
        run_name += '_MAP'
    else:
        run_name += f"_{config['laplace']['laplace_type']}_{config['laplace']['prior_structure']}"
        
    
    use_map = False
    seed = config['seed']
    # excluding the laplace part from the config as we are using the new maglikopt script

    to_log = {
        

        'model_name': config['model_name'],
        'dataset_name': config['dataset_name'],
        'optimizer': config['optimizer'],

        'batch_size': config['batch_size'],
        'num_epochs': config['num_epochs'],
        'lr': config['lr'],
        'weight_decay': config['weight_decay'],
        'seed': seed,
        "marglik_param": {
                "laplace": "MAP" if use_map else config['laplace']['laplace_type'],
                "prior_structure": "MAP" if use_map else config['laplace']['prior_structure'],
            }
        
    }
       
    wandb.init(id = unique_id, name=run_name, project='BNN_Sparse', entity="xxxxxx", config=to_log)
    return run_name



config = {
        "model_name": "Distilbert",
        "dataset_name": "IMDB",
        "optimizer": "adam",
        "batch_size": 64,
        "num_epochs": 20,
        "lr": 2e-5,
        "weight_decay": 0.0,
        "seed": seed,
        "laplace": {
            "laplace_type": "DiagLaplace",
            "prior_structure": "scalar",
            "marglik_frequency": 5,
            "num_epochs_burnin": 21,
            "lr_min": 1e-06,
            "n_hypersteps": 50,
        }
    }
    
run_name_orig = config_Wb(config)

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=256)

train_dataset = train_dataset.map(tokenize_function, batched=True)
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

test_dataset = test_dataset.map(tokenize_function, batched=True)
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Ensure the 'label' key exists
for example in train_dataset:
    assert 'label' in example

collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], collate_fn=collator)
test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], collate_fn=collator)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MyDistilBert(nn.Module):
    """
    Huggingface LLM wrapper for DistilBERT.

    Args:
        tokenizer: The tokenizer used for preprocessing the text data.
    """
    def __init__(self) -> None:
        super().__init__()
        config = DistilBertConfig.from_pretrained('distilbert-base-uncased', num_labels=2)
        self.distilbert = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased', config=config
        )

    def forward(self, data: MutableMapping) -> torch.Tensor:
            
        device = next(self.parameters()).device
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        return output.logits
       

model = MyDistilBert()
model = model.to(device)
# watch the model
wandb.watch(model)


    
la, model, margliks = marglik_optimization(
    model,
    train_dataloader,
    #test_dataloader,
    n_epochs=config["num_epochs"],
    n_epochs_burnin=config["laplace"]["num_epochs_burnin"],
    laplace=DiagLaplace,
    prior_structure='diagonal',
    lr=config['lr'],
    log_wandb=True,
)

num_classes_brier = 2
tunemethod = "map"
config['tune'] = True # one shot
sparsities = [20,40,60,70,75,80,85,90,95,99]
args_sparse= {
    'num_classes': num_classes_brier,
    'prior_structure': config['laplace']['prior_structure'],
    'tune_epochs_burnin': 11 if tunemethod == "map" else 0,
    'marglik_frequency': config['laplace']['marglik_frequency'],
    'fine_tune': config["tune"],
    'tune_epochs': 5,
    'lr': config['lr']*0.1
}

sparse_list = {'laplacekron':{'model': copy.deepcopy(model), 'function': sparse_strategy ,'sparsities': sparsities, 'la': la},
                'magnitude':{'model': copy.deepcopy(model), 'function': magnitude_pruning ,'sparsities': sparsities, 'la': la},
                'random':{'model': copy.deepcopy(model), 'function': random_strategy ,'sparsities': sparsities, 'la': la},
                'SNIP':{'model': copy.deepcopy(model), 'function': SNIP_strategy ,'sparsities': sparsities, 'la': la},
                'GraSP':{'model': copy.deepcopy(model), 'function': GraSP_strategy ,'sparsities': sparsities, 'la': la},
                }

    
for sparse_name, sparse_dict in sparse_list.items():
    cfgs = config
    run_name = run_name_orig
    run_name = f"{run_name}_sub_{sparse_name}_finetune_{args_sparse['tune_epochs']}_{tunemethod}" if args_sparse['fine_tune'] == True else f"{run_name}_asdl_sub_{sparse_name}"
    
    with wandb.init(reinit=True, id=wandb.util.generate_id(),config = cfgs, project='BNN_Sparse', entity="xxxxxx", name=run_name):
        wandb.config.update(args_sparse, allow_val_change=True)
        wandb.config.update({'sparsification_method': sparse_name}, allow_val_change=True)
        models_stats = sparse_dict['function'](la=sparse_dict['la'],model = sparse_dict['model'], test_loader= test_dataloader,
                                train_loader = train_dataloader, sparsities = sparse_dict['sparsities'],args= args_sparse)
   



    


