
cd /nfs/homedirs/xxxxx/BNNS/experiment/src

configs1=$1
configs2=$2

python experiment_post.py --config_dir=cfgs/WideResNet_newpriors/$configs1
python experiment_post.py --config_dir=cfgs/WideResNet_newpriors/$configs2
