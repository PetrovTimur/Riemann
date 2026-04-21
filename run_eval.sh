#!/bin/bash

export PYTHONPATH=.
HYDRA_FULL_ERROR=1 python training/run_eval.py \
    ++checkpoint_path=/home/timurpetrov/code/riemann/outputs/2026-04-15/09-29-52/checkpoints/ckpt_epoch_0009.pt ++module.model.weights=True ++eval_output_table=try.csv
