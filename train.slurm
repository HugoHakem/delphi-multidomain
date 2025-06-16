#!/bin/bash
#SBATCH --job-name=delphi_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=05:00:00
#SBATCH --gpus=a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

source ~/.bashrc
conda activate delphi

cd /homes/bonazzola/Delphi

python train_hla.py config/train_delphi-hla.py -x delphi-hla-2digits -d --dtype=float16 "$@"

# python train_hla.py config/train_delphi.py --no-hla -x delphi-no-hla2 "$@"
# python train_hla.py config/train_delphi-hla.py --n_embd=240 --dtype=float16 --max_steps=2000 --n_layer=6 --show-config
