logfile="error_log.txt"

# Clear the log file before running the scripts
> $logfile

# Change to the correct directory
cd /nfs/homedirs/xxxxx/BNNS/experiment/src

# Use the absolute path to the Python interpreter
python experiment_online.py --config_dir=cfgs/diag_unitwise --sparsewith=post_and_weights #-t #>> $logfile 2>&1 || echo "Script 1 failed. Moving to the next script." | tee -a $logfile
python experiments3.py --config_dir=cfgs/diag_unitwise2  #-t #>> $logfile 2>&1 || echo "Script 2 failed. Moving to the next script." | tee -a $logfile