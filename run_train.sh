#!/bin/bash
#++train_table=datasets/riemann_invariants_sym/riemann_invariants_only_train_filtered.csv \
#++val_table=datasets/riemann_invariants_sym/riemann_invariants_only_val.csv

export PYTHONPATH=.
HYDRA_FULL_ERROR=1 torchrun --standalone training/run_train.py \
    ++module.model.weights=True ++optimizer.weight_decay=1e-1

