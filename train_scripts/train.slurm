#!/bin/bash
#SBATCH --job-name=delphi_train_nohla_120
#SBATCH --output=logs/delphi_train_dim120/%x_%j.out
#SBATCH --time=3:00:00
#SBATCH --gpus=h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

source ~/.bashrc
conda activate delphi

cd /homes/bonazzola/Delphi

DTYPE_OPTION="--dtype=float32"
DATE=$(date +"%H:%M:%S") 

# EXPNAME="delphi-hla-2digits"
# EXPNAME="delphi-hla-2digits-5fold"
EXPNAME="delphi-nohla-dim120-5fold"
# EXPNAME="delphi-nohla-5fold"

# CONFIG="config/train_delphi-hla4digits.py"
CONFIG="config/train_delphi.py"
HLA_FLAG="--no-hla"
# HLA_FLAG=""

# python train_hla.py config/train_delphi-hla.py -x $EXPNAME -d $DTYPE_OPTION "$@"
# python train_hla.py config/train_delphi.py -x $EXPNAME -d $DTYPE_OPTION "$@"
# python train_hla.py config/train_delphi.py --no-hla -x delphi-no-hla2 "$@"
# python train_hla.py config/train_delphi-hla.py --n_embd=240 --dtype=float16 --max_steps=2000 --n_layer=6 --show-config
# python train_hla_5fold.py $CONFIG --no-hla -x $EXPNAME -d $DTYPE_OPTION "$@"

python train_hla_5fold.py $CONFIG $HLA_FLAG -x $EXPNAME -d $DTYPE_OPTION "$@" 
