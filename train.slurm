#!/bin/bash
#SBATCH --job-name=delphi_train_2digit
#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.err
#SBATCH --time=12:00:00
#SBATCH --gpus=a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

source ~/.bashrc
conda activate delphi

cd /homes/bonazzola/Delphi

DTYPE_OPTION="--dtype=float32"
EXPNAME="delphi-hla-2digits"
DATE=$(date +"%H:%M:%S") 

python train_hla.py config/train_delphi-hla.py -x $EXPNAME -d $DTYPE_OPTION "$@"

# python train_hla.py config/train_delphi.py --no-hla -x delphi-no-hla2 "$@"
# python train_hla.py config/train_delphi-hla.py --n_embd=240 --dtype=float16 --max_steps=2000 --n_layer=6 --show-config
