cd /nfs/homedirs/xxxxx/BNNS/experiment/src

#python experiment_structured.py --config_dir=cfgs/Lenetmnist_newpriors/ --repeat=2  #>> $logfile 2>&1 || echo "Script 1 failed. Moving to the next script." | tee -a $logfile
python experiment_structured_row.py --config_dir=cfgs/mnist_fc/ --repeat=2
python experiment_structured_row.py --config_dir=cfgs/Lenetmnist_newpriors/ --repeat=2