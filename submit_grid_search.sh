#!/bin/bash

seeds=(42 142)
n_embd_vals=(120 240 480)
n_layer_vals=(6 12)
batch_size_vals=(128 256)
learning_rate_vals=(1e-4 3e-4)

for seed in "${seeds[@]}"; do
  for embd in "${n_embd_vals[@]}"; do
    for layer in "${n_layer_vals[@]}"; do
      for batch in "${batch_size_vals[@]}"; do
        for lr in "${learning_rate_vals[@]}"; do
            sbatch train.slurm --n_embd=$embd --n_layer=$layer --batch_size=$batch --learning_rate=$lr --seed=$seed --min_lr=$(awk "BEGIN {print $lr / 10}")
        done
      done
    done
  done
done

