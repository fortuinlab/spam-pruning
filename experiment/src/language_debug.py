import logging
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling, GPT2TokenizerFast
from datasets import load_dataset
import numpy as np
import wandb
import random
from tqdm import tqdm
from pathlib import Path
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, DataCollatorForLanguageModeling, GPT2TokenizerFast, PreTrainedTokenizer
from laplace import Laplace, DiagLaplace
from collections import UserDict
from typing import MutableMapping
from marglikopt_llm import marglik_optimization
from laplace.curvature import AsdlGGN, AsdlEF
# Logging setup
logging.basicConfig(level=logging.INFO)

# Set random seeds for reproducibility
seed = np.random.randint(0, 1000)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# WandB setup
unique_id = wandb.util.generate_id()
run_name = "GPT2_LM_Wikitext"
wandb.init(id=unique_id, name=run_name, project='BNN_Sparse', entity="xxxxxx")

# Set up device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_id = 'gpt2'

# Load the model and tokenizer
class GPT2Wrapper(nn.Module):
    """
    Huggingface GPT-2 wrapper for language modeling.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        super().__init__()
        config = GPT2Config.from_pretrained('gpt2')
        config.pad_token_id = tokenizer.pad_token_id
        self.hf_model = GPT2LMHeadModel.from_pretrained('gpt2', config=config)

    def forward(self, data: MutableMapping) -> torch.Tensor:
        device = next(self.parameters()).device
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        outputs = self.hf_model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        logits = outputs.logits
        if logits.dim() == 3:
            logits = logits.reshape(-1, logits.size(-1))  # Flatten logits to 2D for loss computation
        return logits
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
tokenizer.pad_token_id = tokenizer.eos_token_id   
model = GPT2Wrapper(tokenizer)
#model = nn.DataParallel(model)
model = model.to(device)


# Load and prepare dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
train_dataset = dataset['train']
test_dataset = dataset['test']

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

# Apply tokenization to the datasets
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
train_dataset.set_format(type='torch', columns=['input_ids'])

# Filter out empty sequences
def filter_empty_sequences(example):
    return len(example['input_ids']) > 0

train_dataset = train_dataset.filter(filter_empty_sequences)

# Data collator and data loaders
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
train_dataloader = DataLoader(train_dataset, batch_size=8, collate_fn=collator, shuffle=True)  # Adjusted batch size

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss(reduction='mean')

# sttochastic = Ture check 

"""la, model, margliks, losses = marglik_optimization(
    model=model, train_loader=train_dataloader, likelihood='language_modelling',
    laplace= DiagLaplace, n_epochs=1, backend = AsdlEF,
     prior_structure='diagonal'
)"""

# Training loop
train = True
if train:
    model.train()
    for epoch in range(1): 
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            # Shift labels for language modeling
            labels = input_ids.clone()
            outputs = model(batch)
            loss = criterion(outputs, labels.view(-1))
            loss.backward()
            optimizer.step()
            wandb.log({"train_loss": loss.item(), "epoch": epoch})
            
la = Laplace(model,
    likelihood='classification',
    subset_of_weights='all',
    hessian_structure='diag',
    backend=AsdlEF,)
    #backeend_kwargs= {"stochastic":True}
la.fit(train_dataloader)

# Save the model
# path_save = Path(__file__).parent.parent.parent.parent.parent.parent / "xxxxxx/gpt2_finetuned"
# model.save_pretrained(path_save)
# tokenizer.save_pretrained(path_save)

# Evaluation with Perplexity
"""model.eval()
encodings = tokenizer('\n\n'.join(test_dataset['text']), return_tensors='pt')

# Define max length and stride
max_length = model.hf_model.config.n_positions
stride = 512

# List to store log likelihoods
lls = []

# Sliding window evaluation
for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
    begin_loc = max(i + stride - max_length, 0)
    end_loc = min(i + stride, encodings.input_ids.size(1))
    trg_len = end_loc - i  # May be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100  # Mask context tokens

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        log_likelihood = outputs.loss * trg_len

    lls.append(log_likelihood)

# Calculate perplexity
ppl = torch.exp(torch.stack(lls).sum() / end_loc)

print(f"Perplexity: {ppl.item()}")
wandb.log({"test_loss": log_likelihood, "perplexity": ppl.item()})
"""
wandb.finish()
