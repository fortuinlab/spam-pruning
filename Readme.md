# Shaving Neural Network Weights with Occam's Razor

## Getting Started
This repository is an implementation of the paper  "Shaving Weights with Occam’s Razor:Bayesian Sparsification for Neural Networks using the Marginal Likelihood". The paper is available on [arXiv](https://arxiv.org/abs/2402.15978).

### Prerequisites
To run the experiments, you will need to install our edited version of the public Laplace-torch library, contained under the [Laplace_kfac_diag_unitwise](experiment/src/Laplace_kfac_diag_unitwise). This library includes diagonal and unitwise priors and handles specific types of dataloaders and pre-trained models.

### Running the Experiments
The `cfgs` folder contains the configuration files for the experiments. The `generate_config.py` script can be used to generate new configuration files. We provide a few examples of configuration files in the `cfgs` folder. We recommend using the `generate_config.py` script to generate new configuration files based on the desired experiment or to relocate some in a new directory.

To run the experiments, you can use the following scripts located in the [experiment/src](experiment/src) folder:

- [`experiment_post`](experiment/src/experiment_post.py) for running unstructured pruning experiments with the following arguments:
  - `--config_dir`: Path to the configuration folder
  - `--repeat`: Number of times to repeat the experiment for each configuration file (different seeds)
  - `--tune`: A flag to indicate if we want to tune the model after pruning or not (default is False)
  - `--mode`: `posthoc` or `online` (default is `posthoc`). We provide a separate online pruning script for the online pruning experiments

- [`experiment_online`](experiment/src/experiment_online.py) for running online pruning experiments with SpaM OPD with the following arguments:
  - `--config_dir`: Path to the configuration folder
  - `--repeat`: Number of times to repeat the experiment for each configuration file (different seeds)
  - `--sparsewith`: Takes either `post_only` or `post_and_weights` to indicate the type of scoring to be used. Default is `post_and_weights`, which is the OPD scoring. This was used for an ablation study to see the posterior quality and the effect of the posterior and the combination of the posterior and the weight's magnitude.

- [`experiment_structured_all`](experiment/src/experiment_structured_all.py) for running the structured pruning experiments. This uses the same parameters as the `experiment_post` script.

- [`experiment_semi`](experiment/src/experiment_semi.py) for pruning using the 2:4 scheme. We do not feature the results in the paper as this approach maintains the full baseline performance in the majority of cases.

### Important Scripts

- [`generate_config.py`](experiment/src/generate_config.py): Script to generate the configuration files for the experiments.
- [`marglikopt.py`](experiment/src/marglikopt.py): Script to run SpaM and MAP training with different priors (MAP is when the passed `num_epochs_burning` is greater than `num_epochs`).
- [`marglikopt_sp_new.py`](experiment/src/marglikopt_sp_new.py): Online pruning during SpaM and dynamic masking.
- [`sparsify_v2.py`](experiment/src/sparsify_v2.py): Unstructured pruning criteria.
- [`structured_sparsity.py`](experiment/src/structured_sparsity.py): Utilities for structured pruning and `scoring` contains the scoring function. Example of usage:
    ```python
    score = prune_scoring[method](model_copy, self.train_loader, self.config['device'], self.criterion)
    mask_dict, neurons_pruned, filters_pruned = structured_prune(model_copy, score, prune_percentage)
    ```

- [`result_notebook_fn_fashion.ipynb`](experiment/src/result_notebook_fn_fashion.ipynb): A notebook that illustrates the results of the structured compression and deletion of neurons and filters.

### Self-Contained Scripts for Fast Experiments
- [`gpt2_diag.py`](experiment/src/gpt2_diag.py): Script to run the GPT2 experiments with diagonal Laplace and parameterwise prior.
- [`gpt2.py`](experiment/src/gpt2.py): GPT2 experiments with MAP.
- [`vit.py`](experiment/src/vit.py): Scratch ViT on MNIST.
- [`vit_cifar100`](experiment/src/vit_cifar100.py): Scratch ViT on CIFAR100.
- [`bert_marg_diag.py`](experiment/src/bert_marg_diag.py): DistilBERT experiments with diagonal Laplace and parameterwise prior.
- [`bert_marg.py`](experiment/src/bert_marg.py): DistilBERT experiments with MAP.
- [`vit_cifar100_transfer_diag.py`](experiment/src/vit_cifar100_transfer_diag.py): ViT on CIFAR100 transfer learned from pre-trained ViT on ImageNet with diagonal Laplace and parameterwise prior.

## Note
We are currently working on cleaning up the code and adding notebook tutorial for the experiments. 
