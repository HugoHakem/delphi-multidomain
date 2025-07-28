#!/bin/bash

seeds=(142)
n_embd_vals=(120)
n_layer_vals=(6)
batch_size_vals=(128)
learning_rate_vals=(1e-4)
val_folds=(1 2 3 4 5)

for seed in "${seeds[@]}"; do
  for embd in "${n_embd_vals[@]}"; do
    for layer in "${n_layer_vals[@]}"; do
      for batch in "${batch_size_vals[@]}"; do
        for lr in "${learning_rate_vals[@]}"; do
          for val_fold in "${val_folds[@]}"; do
            sbatch train.slurm --val_fold $val_fold --n_embd=$embd --n_layer=$layer --batch_size=$batch --learning_rate=$lr --seed=$seed --min_lr=$(awk "BEGIN {print $lr / 10}")
            sleep 5
          done
        done
      done
    done
  done
done
