#!/bin/bash

HLA_ALLELE_AGGREGATION="1_field"
HLA_ALLELE_AGGREGATION="2_fields"
HLA_ALLELE_AGGREGATION="peptide_pseudosequence"

DISEASE_AGGREGATION="2_levels"

ATTENTION_SCHEME="within_gene within_hla all all all all"

CONFIG=config/train_delphi-hla.py

python new_train.py \
  --config $CONFIG \
  --batch_size 64 \
  --block_size 64 \
  --domains diseases sex hla_alleles lifestyle death \
  --domain_aggregation_schemes hla_alleles=2_digits\
  --attention_segregation \
  --n_head 10 \
  --n_embd 120 \
  --learning_rate 6e-4 \
  --min_learning_rate 6e-5 \
  --warmup_iters 1000 \
  --no_event_token_rate 5 \
  --log_intervals 100 \
  --seed 42 \
  --dry-run \
  --compute_auc \
  -x $EXPNAME \
  -d $DTYPE_OPTION
