import argparse
import json
import os
from pathlib import Path
import re
import random
import pandas as pd
from train_cfg_pai import train_cfg

def get_config_from_json(json_file):
    with open(json_file) as f:
        config = json.load(f)
    return config

def process_config_file(file, json_dir,args_dict):
    
    config = get_config_from_json(json_dir / file)
    # add the arguments to the config
    for key, value in args_dict.items():
        config[key] = value
    print(config)
    a = train_cfg(config, file) 
    a.config_Wb()
    a.Dataloader()
    a.load_encoder()
    a.train()
    a.finish_Wb()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', type=str, default="cfgs/LeNet/",
                        help='Directory containing config files')

    parser.add_argument('--mode', choices=['posthoc', 'online'], default='posthoc',
                        help='Training mode: posthoc or online')
    parser.add_argument('-t', '--tune', action='store_true',default=False,
                        help='Enable tuning in posthoc mode')
    
    parser.add_argument('--tuning_method', choices=['same', 'map'], default='map',
                        help='Tuning method: ')
    parser.add_argument('--repeat', type=int, default=4,
                            help='Number of times to repeat each experiment')
    
    parser.add_argument('--skip_method', choices=['none', 'SNIP','magnitdue',"random"], default='none',
                        help='Skip method experiment ')
    
    args = parser.parse_args()
    args_dict = vars(args) #
    # make sure to use the right path for the set of experiments
    default = False 
    repeat = args.repeat
    json_dir = Path(__file__).parent.parent / args.config_dir

    if not json_dir.exists():
        print(f"Directory {json_dir} does not exist.")
        exit(1)

    json_files = [
        pos_json
        for pos_json in os.listdir(json_dir)
        if pos_json.endswith(".json") and not pos_json.startswith("default")
        ]

    if default:
        # We want to make sure that the configs are processed in their numerical order
        JSON_FILE_PREFIX = "experiment_"
        json_files.sort(
            key=lambda var: [
                int(x) if x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)
            ]
        )

        # Enter all numbers which should be excluded:
        skip_experiments = []
        for file in json_files:
            if file.startswith(JSON_FILE_PREFIX):
                experiment_key = int(os.path.splitext(file)[0][len(JSON_FILE_PREFIX) :])
            if experiment_key in skip_experiments:
                continue

            process_config_file(file, json_dir,args_dict)
    else:
        
        for _ in range(repeat):
            for file in json_files:
                process_config_file(file, json_dir,args_dict)