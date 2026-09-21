#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<mahpe@dtu.dk>  # The default value is the submitting user.
#SBATCH --partition=h200 #
#SBATCH --job-name=space_group_split
#SBATCH --output=space_group_split.out
#SBATCH --error=space_group_split.err
#SBATCH -N 1-1
#SBATCH -n 24 #
#SBATCH --gres=gpu:H200:1 #
#SBATCH --time=50:00:00 # 1 day, 2 hours, 0 min, 0 sec.
#SBATCH --begin=now+0hour # 0 is in seconds
##SBATCH --exclusive
#SBATCH --mem-per-gpu=150G

nvidia-smi

module use /home/energy/modules/modules/all

conda activate defect_MLIP

python Mace_run_train.py --cfg Mace.toml --file_name space_group_split --root_dir ./datasplit/
