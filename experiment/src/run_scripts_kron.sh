logfile="error_log.txt"

# Clear the log file before running the scripts
> $logfile

# Change to the correct directory
cd /nfs/homedirs/xxxxx/BNNS/experiment/src

# Use the absolute path to the Python interpreter
  #>> $logfile 2>&1 || echo "Script 1 failed. Moving to the next script." | tee -a $logfile
python experiment_online.py --config_dir=cfgs/missingonline_bt/bt3 --sparsewith=post_and_weights

