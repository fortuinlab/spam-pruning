import json
import os
# displaying the default configuration here so we can easily update and expand it
# note that all passable parameters even the ones that are recommended to be fixed will be displayed here

DEFAULT_CONFIG = {
    "seed": 42, # seed if fixed internally in the training script and updated in wandb
    "dataset_name": "breast_cancer",
    "model_name": "CancerNet_fc",
    "lr": 0.001,
    "weight_decay": 0.0,
    "batch_size": 128,
    "num_epochs": 50,
    "use_map": False,
    "train_marglik": True,
    "optimizer": "adam",
    "criterion": "cross_entropy",
    "hessian_structure": "full",
    "device": "cuda",
    "sparse_method": None,  # Add the sparse_method parameter here
    "laplace": {
        "backend": "BackPackGGN",
        "n_epochs_burnin": 0, # the default for complexed structure we set it to 10 or  20
        "hessian_structure": "full",
        "la_optimizer_kwargs": {
            "lr": 0.01
        },
        "lr_min": None,
        "laplace_type": "KronLaplace",
        "prior_structure": "layerwise",
        "marglik_frequency": 1,
        "n_hypersteps": 100,
        
    },
    "wandb": {
        "project": "BNN_Sparse",
        "entity": "xxxxxx"
    }
}

EXPORT_DIR = "experiment/cfgs/"


def config_generator(**kwargs):
    """Generates a configuration dictionary based on the default values
    and optional keyword arguments.

    Args:
        **kwargs: Optional keyword arguments to update the default configuration.

    Returns:
        A dictionary representing the final configuration.
    """
    config = DEFAULT_CONFIG.copy()
    config.update(kwargs)
    return config


def write_config(config, filename):
    """Writes a configuration dictionary to a JSON file.

    Args:
        config (dict): The configuration dictionary.
        filename (str): The name of the output JSON file.
    """
    with open(filename, "w") as f:
        json.dump(config, f, indent=4)


def generate_configs(dataset_name, model_name, export_dir):
    laplace_type = ["KronLaplace", "DiagLaplace"]
    prior_structure = ["layerwise", "diagonal"]
    sparse_methods = [None, "random", "magnitude"]

    for i in range(len(laplace_type)):
        for j in range(len(prior_structure)):
            for k in range(len(sparse_methods)):
                dict = { 
                "dataset_name": dataset_name,
                "model_name": model_name,
                "sparse_method": sparse_methods[k],
                    "laplace": {
                    "laplace_type": laplace_type[i],
                    "prior_structure": prior_structure[j],
                    "lr_min": None,
                    "marglik_frequency": 1,
                    "n_hypersteps": 100,
                    "lr": 0.001
                }}
                config = config_generator(**dict)
                write_config(config, export_dir + f"{model_name}_{dataset_name}_{laplace_type[i]}_{prior_structure[j]}.json")

def generate_configs_train_once(dataset_name, model_name, export_dir):
    # Refine this based on your needs
    laplace_type = ["KronLaplace", "DiagLaplace"]
    prior_structure = ["layerwise", "diagonal", "scalar", "unitwise"]
    #sparse_methods = [None, "random", "magnitude"]

    for i in range(len(laplace_type)):
        for j in range(len(prior_structure)):
            #for k in range(len(sparse_methods)):
            dict = { 
            "dataset_name": dataset_name,
            "model_name": model_name,
            "epochs": 50,
            "lr": 0.001,
            "batch_size": 64,
            "optimizer": "adam",
            #"sparse_method": sparse_methods[k],
                "laplace": {
                "laplace_type": laplace_type[i],
                "prior_structure": prior_structure[j],
                "lr_min": None,
                "marglik_frequency": 1,
                "n_hypersteps": 100,
                "lr": 0.001,
                "n_epochs_burnin": 0,
            }}
            config = config_generator(**dict)
            # additional map config where only the n_epochs_burnin is updated with epochs +1
            dict_map = dict.copy()
            dict_map["use_map"] = True
            dict_map["laplace"]["n_epochs_burnin"] = dict["epochs"] + 1
            dict_map["laplace"]["laplace_type"] = "DiagLaplace"
            dict_map["laplace"]["prior_structure"] = "scalar"
            config_map = config_generator(**dict_map)
            write_config(config_map, export_dir + f"{model_name}_{dataset_name}_MAP.json")

            write_config(config, export_dir + f"{model_name}_{dataset_name}_{laplace_type[i]}_{prior_structure[j]}_.json")


def generate_configs_lenet(dataset_name, model_name, export_dir):
    laplace_type = ["KronLaplace", "DiagLaplace"]
    prior_structure = ["layerwise", "diagonal", "scalar","unitwise"]
    #sparse_methods = [None, "random", "magnitude"]

    for i in range(len(laplace_type)):
        for j in range(len(prior_structure)):

            #for k in range(len(sparse_methods)):
            dict = { 
            "dataset_name": dataset_name,
            "model_name": model_name,
            "lr": 0.1,
            "use_map": False,
            "num_epochs": 100,
            "batch_size": 128,
            "optimizer": "sgd",
            #"sparse_method": sparse_methods[k],
                "laplace": {
                "laplace_type": laplace_type[i],
                "prior_structure": prior_structure[j],
                "lr": 0.1,
                "lr_hyp": 0.1,
                "lr_min": 1e-6,
                "marglik_frequency": 5,
                "n_hypersteps": 100,
                "n_epochs_burnin": 0,


            }}
            config = config_generator(**dict)
            write_config(config, export_dir + f"{model_name}_{dataset_name}_{laplace_type[i]}_{prior_structure[j]}_.json")



def generate_configs_vit(dataset_name, model_name, export_dir):
    laplace_type = ["KronLaplace", "DiagLaplace"]
    prior_structure = ["layerwise", "diagonal", "scalar"]
    #sparse_methods = [None, "random", "magnitude"]

    for i in range(len(laplace_type)):
        for j in range(len(prior_structure)):
            #for k in range(len(sparse_methods)):
            dict = { 
            "dataset_name": dataset_name,
            "model_name": model_name,
            "lr": 0.001,
            "use_map": False,
            "num_epochs": 100,
            "batch_size": 128,
            "optimizer": "adam",
            #"sparse_method": sparse_methods[k],
                "laplace": {
                "laplace_type": laplace_type[i],
                "prior_structure": prior_structure[j],
                "lr": 0.1,
                "lr_hyp": 0.1,
                "lr_min": 1e-6,
                "marglik_frequency": 5,
                "n_hypersteps": 100,
                "n_epochs_burnin": 0,


            }}
            config = config_generator(**dict)
            write_config(config, export_dir + f"{model_name}_{dataset_name}_{laplace_type[i]}_{prior_structure[j]}_.json")

def generate_configs_mixers(dataset_name, model_name, export_dir):
    laplace_type = ["KronLaplace", "DiagLaplace"]
    prior_structure = ["layerwise", "diagonal", "scalar"]
    #sparse_methods = [None, "random", "magnitude"]

    for i in range(len(laplace_type)):
        for j in range(len(prior_structure)):
            if laplace_type[i] == "KronLaplace" and prior_structure[j] == "diagonal":
                continue
            #for k in range(len(sparse_methods)):
            dict = { 
            "dataset_name": dataset_name,
            "model_name": model_name,
            "lr": 0.001,
            "use_map": False,
            "num_epochs": 100,
            "batch_size": 128,
            "optimizer": "adam",
            #"sparse_method": sparse_methods[k],
                "laplace": {
                "laplace_type": laplace_type[i],
                "prior_structure": prior_structure[j],
                "lr": 0.1,
                "lr_hyp": 0.1,
                "lr_min": 1e-6,
                "marglik_frequency": 5,
                "n_hypersteps": 100,
                "n_epochs_burnin": 0,


            }}
            config = config_generator(**dict)
            write_config(config, export_dir + f"{model_name}_{dataset_name}_{laplace_type[i]}_{prior_structure[j]}_.json")


def generate_configs_train_once_resnet(dataset_name, model_name, export_dir):
    laplace_type = ["KronLaplace", "DiagLaplace"]
    prior_structure = ["layerwise", "diagonal", "scalar", "unitwise"]
    #sparse_methods = [None, "random", "magnitude"]

    for i in range(len(laplace_type)):
        for j in range(len(prior_structure)):
            #for k in range(len(sparse_methods)):
            dict = { 
            "dataset_name": dataset_name,
            "model_name": model_name,
            "lr": 0.1,
            "use_map": False,
            "num_epochs": 100,
            "resnet_inplanes": 64,
            "batch_size": 128,
            "optimizer": "sgd",
            #"sparse_method": sparse_methods[k],
                "laplace": {
                "laplace_type": laplace_type[i],
                "prior_structure": prior_structure[j],
                "lr": 0.1,
                "lr_hyp": 0.1,
                "lr_min": 1e-6,
                "marglik_frequency": 5,
                "n_hypersteps": 50,
                "n_epochs_burnin": 20,
                


            }}
            config = config_generator(**dict)
            write_config(config, export_dir + f"{model_name}_{dataset_name}_{laplace_type[i]}_{prior_structure[j]}_.json")




def generate_configs_train_wideresnet(dataset_name, model_name, export_dir):
    laplace_type = ["KronLaplace", "DiagLaplace"]
    prior_structure = ["layerwise", "diagonal", "scalar", "unitwise"]


    for i in range(len(laplace_type)):
        for j in range(len(prior_structure)):

            dict = { 
            "dataset_name": dataset_name,
            "model_name": model_name,
            "lr": 0.1,
            "use_map": False,
            "num_epochs": 100,
            "resnet_inplanes": 64,
            "batch_size": 128,
            "optimizer": "sgd",
            "decay": "cos",
                "laplace": {
                "laplace_type": laplace_type[i],
                "prior_structure": prior_structure[j],
                "temperature": 5,
                "lr": 0.1,
                "lr_hyp": 0.1,
                "lr_min": 1e-6,
                "marglik_frequency": 5,
                "n_hypersteps": 100,
                "n_epochs_burnin": 20,
                

            }}
            config = config_generator(**dict)
            write_config(config, export_dir + f"{model_name}_{dataset_name}_{laplace_type[i]}_{prior_structure[j]}_.json")








def generate_configs_train_once_resnet_map(dataset_name, model_name, export_dir):
    laplace_type = ["DiagLaplace"]
    prior_structure = ["scalar"]
    #sparse_methods = [None, "random", "magnitude"]

    for i in range(len(laplace_type)):
        for j in range(len(prior_structure)):
            if laplace_type[i] == "KronLaplace" and prior_structure[j] == "diagonal":
                continue
            #for k in range(len(sparse_methods)):
            dict = { 
            "dataset_name": dataset_name,
            "model_name": model_name,
            "lr": 0.1,
            "use_map": True,
            "num_epochs": 100,
            "resnet_inplanes": 64,
            "batch_size": 128,
            "optimizer": "sgd",
            #"sparse_method": sparse_methods[k],
                "laplace": {
                "laplace_type": laplace_type[i],
                "prior_structure": prior_structure[j],
                "lr": 0.1,
                "lr_hyp": 0.1,
                "lr_min": 1e-6,
                "marglik_frequency": 5,
                "n_hypersteps": 100,
                "n_epochs_burnin": 101,
                'prior_prec_init': 1.
                


            }}
            config = config_generator(**dict)
            write_config(config, export_dir + f"{model_name}_{dataset_name}_{laplace_type[i]}_{prior_structure[j]}_map_sparse_once.json")


def generate_configs_imagenet_ResNet50(dataset_name, model_name, export_dir):
    laplace_type = ["DiagLaplace"]
    prior_structure = [ "diagonal"]
    #sparse_methods = [None, "random", "magnitude"]

    for i in range(len(laplace_type)):
        for j in range(len(prior_structure)):
            #for k in range(len(sparse_methods)):
            dict = { 
            "dataset_name": dataset_name,
            "model_name": model_name,
            "lr": 0.1,
            "use_map": False,
            "num_epochs": 100,
            "resnet_inplanes": 64,
            "batch_size": 128,
            "optimizer": "sgd",
            #"sparse_method": sparse_methods[k],
                "laplace": {
                "laplace_type": laplace_type[i],
                "prior_structure": prior_structure[j],
                "temperature": 5,
                "lr": 0.1,
                "lr_hyp": 0.1,
                "lr_min": 1e-6,
                "marglik_frequency": 5,
                "n_hypersteps": 100,
                "n_epochs_burnin": 20,
            
            }}
            config = config_generator(**dict)
            write_config(config, export_dir + f"{model_name}_{dataset_name}_{laplace_type[i]}_{prior_structure[j]}_.json")



def main():
    dataset_names = ["mnist","FashionMnist","Cifar10"]
    model_names = ["LeNet","LeNet","LeNet"]
    #model_names = ["CancerNet_fc"]
    from pathlib import Path
    
    export_dirs = ["../cfgs/Lenetmnist_newpriors/", "../cfgs/LeNetfashionmnist_newpriors/", "../cfgs/LeNetcifar10_newpriors/"]

    assert len(dataset_names) == len(model_names) == len(export_dirs), "All lists should have the same length."

    for i, (dataset_name, model_name, export_dir) in enumerate(zip(dataset_names, model_names, export_dirs)):
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        #generate_configs(dataset_name, model_name, export_dir)
        generate_configs_lenet(dataset_name, model_name, export_dir)



def main2():
    dataset_names = ["Cifar10"]
    model_names = ["MLPMixer"]
    export_dirs = ["experiment/cfgs/MLPMixerCifar10/"]

    assert len(dataset_names) == len(model_names) == len(export_dirs), "All lists should have the same length."

    for i, (dataset_name, model_name, export_dir) in enumerate(zip(dataset_names, model_names, export_dirs)):
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        #generate_configs_train_once_resnet(dataset_name, model_name, export_dir)
        #generate_configs_train_wideresnet(dataset_name, model_name, export_dir)
        generate_configs_mixers(dataset_name, model_name, export_dir)

def main3():
    dataset_names = ["Cifar10"]
    model_names = ["WideResNet"]
    export_dirs = ["../cfgs/WideResNet_newpriors/"]

    assert len(dataset_names) == len(model_names) == len(export_dirs), "All lists should have the same length."

    for i, (dataset_name, model_name, export_dir) in enumerate(zip(dataset_names, model_names, export_dirs)):
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        #generate_configs_train_once_resnet(dataset_name, model_name, export_dir)
        generate_configs_train_wideresnet(dataset_name, model_name, export_dir)
        

def main4(): 
    dataset_names = ["mnist"]
    model_names = ["ViT"]
    export_dirs = ["../cfgs/vit_mnist/"]

    assert len(dataset_names) == len(model_names) == len(export_dirs), "All lists should have the same length."

    for i, (dataset_name, model_name, export_dir) in enumerate(zip(dataset_names, model_names, export_dirs)):
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        #generate_configs_train_once_resnet(dataset_name, model_name, export_dir)
        generate_configs_vit(dataset_name, model_name, export_dir)





if __name__ == "__main__":
    #main()
    main3()
    #main2()
    #main2()
    #main4()
    # Make sure you update generate_configs_train_once parameters based on your needs !!!
    """dataset_names = ["mnist","breast_cancer"]
    #dataset_names = ["Cifar10"]
    #dataset_names = ["breast_cancer"]
    model_names = ["MNIST_FC", "CancerNet_fc"]
    #model_names = ["Cifar10_CNN"]
    #model_names = ["CancerNet_fc"]
    export_dirs = ["../cfgs/MNIST_FC_newpriors/", "../cfgs/CancerNet_fc_newpriors/"]

    assert len(dataset_names) == len(model_names) == len(export_dirs), "All lists should have the same length."

    for i, (dataset_name, model_name, export_dir) in enumerate(zip(dataset_names, model_names, export_dirs)):
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        #generate_configs(dataset_name, model_name, export_dir)
        generate_configs_train_once(dataset_name, model_name, export_dir)"""