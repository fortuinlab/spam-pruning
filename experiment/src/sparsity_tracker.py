import torch
from pathlib import Path
import os
import sys
import matplotlib.pyplot as plt


from models import CancerNet_fc, BostonReg_fc, ResNet18
sys.path.append(str(Path(__file__).parent.parent.parent / "overparametrized_fc"))

from src.model_classification import *
from src.utils import *
from src.dataloader import *

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"

def data_loaders():
    # Load data
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

    train_dataset = CancerDataset_supported(X_train, y_train)
    test_dataset = CancerDataset_supported(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    return train_loader, test_loader

def check_sparsity_cfg_runs(model_path, export_full=False):
    # Check if the model exists
    if not os.path.exists(model_path):
        print(f"'{model_path}' does not exist.")
        return

    # Load the pickled model
    if export_full:
        model = torch.load(model_path)
    else:
        model = CancerNet_fc(input_size=30, hidden_size=100, output_size=2)
        model.load_state_dict(torch.load(model_path))

    # Calculate sparsity of each layer
    total_params = 0
    total_zero_params = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            num_zero_params = param.eq(0.0).sum().item()
            sparsity = num_zero_params / num_params

            print(f"Layer '{name}': {100*sparsity:.5f}% sparsity")

            total_params += num_params
            total_zero_params += num_zero_params

    # Calculate sparsity of entire model
    print(f"Total parameters: {total_params}")
    print(f"Total zero parameters: {total_zero_params}")
    model_sparsity = total_zero_params / total_params
    print(f"Total model sparsity: {100*model_sparsity:.2f}%")
    _, test_loader = data_loaders()
    acc = eval(model, test_loader, "cuda")
    print(f"Accuracy: {acc:.2f}%")
    print()
    name_only = str(model_path).split("/")[-1]
    name_only = name_only.split(".")[0]
    
    return {"model": name_only, "sparsity": model_sparsity, "accuracy": acc}


def plot_sparsity_accuracy(results):
    models = [str(r["model"]) for r in results]
    sparsities = [r["sparsity"] for r in results]
    accuracies = [r["accuracy"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))  
    ax2 = ax.twinx()

    ax.bar(models, sparsities, color="b")
    ax2.plot(models, accuracies, color="r", marker="o")

    ax.set_xlabel("Model")
    ax.set_ylabel("Sparsity (%)")
    ylabel2 = ax2.set_ylabel("Accuracy (%)")
    ylabel2.set_color("r")  

    ax.tick_params(axis='y', labelcolor="b")
    ax2.tick_params(axis='y', labelcolor="r")

    ax.set_xticklabels(models, rotation=90, fontsize=8)  
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    use_run = False

    runs = os.listdir(RESULTS_DIR)
    model_path_note = Path(__file__).parent.parent.parent / "overparametrized_fc/laplace_models/"
    models = os.listdir(model_path_note)
    
    results = []

    if use_run:
        print(runs)
        for run in runs:
            model_path = RESULTS_DIR / run / "model.pt"
            print(f"Checking sparsity of '{model_path}'...")
            result = check_sparsity_cfg_runs(model_path)
            results.append(result)
    else:
        for model in models:
            if model == "model_diag.pt" or model == "model_kron.pt":
                continue
            model_path = model_path_note / model
            print(model)
            print(f"Checking sparsity of '{model_path}'...")
            result = check_sparsity_cfg_runs(model_path, export_full=True)
            results.append(result)
    
    plot_sparsity_accuracy(results)

