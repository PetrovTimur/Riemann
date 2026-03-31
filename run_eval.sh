#!/bin/bash

export PYTHONPATH=.
HYDRA_FULL_ERROR=1 torchrun --standalone training/run_eval.py \
    ++checkpoint_path=outputs/2026-03-22/14-33-02/checkpoints/ckpt_epoch_0079.pt ++module.model.weights=True ++eval_output_table=try.csv
