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
from transformers import GPT2Tokenizer, GPT2Config, GPT2ForSequenceClassification
from transformers import ( # noqa: E402
    GPT2Config,
    GPT2ForSequenceClassification,
    GPT2Tokenizer,
    DataCollatorWithPadding,
    PreTrainedTokenizer,
)
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


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token_id = tokenizer.eos_token_id
# Load and prepare dataset
dataset = load_dataset("imdb")
train_dataset = dataset['train'].shuffle(seed=42)
test_dataset = dataset['test'].shuffle(seed=42)

# Print a few examples
print(train_dataset[0])

# Tokenization

def config_Wb(config):
    unique_id = wandb.util.generate_id()
    run_name = f"GPT2_IMDB_{config['num_epochs']}"
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
        "model_name": "Gpt2",
        "dataset_name": "IMDB",
        "optimizer": "adam",
        "batch_size": 8,
        "num_epochs": 10,
        "lr": 2e-5,
        "weight_decay": 0.0,
        "seed": seed,
        "laplace": {
            "laplace_type": "DiagLaplace",
            "prior_structure": "scalar",
            "marglik_frequency": 5,
            "num_epochs_burnin": 11,
            "lr_min": 1e-06,
            "n_hypersteps": 50,
        }
    }
    
run_name_orig = config_Wb(config)

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=1024)

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

class MyGPT2(nn.Module):
    """
    Huggingface LLM wrapper.

    Args:
        tokenizer: The tokenizer used for preprocessing the text data. Needed
            since the model needs to know the padding token id.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        super().__init__()
        config = GPT2Config.from_pretrained('gpt2')
        config.pad_token_id = tokenizer.pad_token_id
        config.num_labels = 2
        self.hf_model = GPT2ForSequenceClassification.from_pretrained(
            'gpt2', config=config
        )

    def forward(self, data: MutableMapping) -> torch.Tensor:
        '''
        Custom forward function. Handles things like moving the
        input tensor to the correct device inside.

        Args:
            data: A dict-like data structure with `input_ids` inside.
                This is the default data structure assumed by Huggingface
                dataloaders.

        Returns:
            logits: An `(batch_size, n_classes)`-sized tensor of logits.
        '''
        device = next(self.parameters()).device
        input_ids = data['input_ids'].to(device)
        attn_mask = data['attention_mask'].to(device)
        output_dict = self.hf_model(input_ids=input_ids, attention_mask=attn_mask)
        return output_dict.logits
       

model = MyGPT2(tokenizer)
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
    prior_structure='scalar',
    lr=config['lr'],
    log_wandb=True,
)

num_classes_brier = 2
tunemethod = "map"
config['tune'] = False # one shot
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
    
    try:
        with wandb.init(reinit=True, id=wandb.util.generate_id(),config = cfgs, project='BNN_Sparse', entity="xxxxxx", name=run_name):
            wandb.config.update(args_sparse, allow_val_change=True)
            wandb.config.update({'sparsification_method': sparse_name}, allow_val_change=True)
            models_stats = sparse_dict['function'](la=sparse_dict['la'],model = sparse_dict['model'], test_loader= test_dataloader,
                                    train_loader = train_dataloader, sparsities = sparse_dict['sparsities'],args= args_sparse)
    except Exception as e:
        print(f"An error occurred: {e}")

    


