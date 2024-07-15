#!/bin/bash

SEED=0
ALPHA=0.01

python src/probing.py \
    --model_dir "output/single_turn/alpha${ALPHA}/s${SEED}" \
    --model_dir "output/single_turn/alpha${ALPHA}/s${SEED}/probing" \
    --data_dir "data/single_turn/alpha0.1" \
    --max_examples 1024 \
    --max_rows 1024;