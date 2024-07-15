#!/bin/bash

NUM_EPOCHS=100
BATCH_SIZE=8
LR=0.0001
NUM_HEADS=12
NUM_LAYERS=8
HIDDEN_SIZE=768
SEED=0

python src/run.py \
    --data_dir "data/multi_turn" \
    --num_layers "${NUM_LAYERS}" \
    --num_heads "${NUM_HEADS}" \
    --hidden_size "${HIDDEN_SIZE}" \
    --lr "${LR}" \
    --num_epochs "${NUM_EPOCHS}" \
    --seed "${SEED}" \
    --train_batch_size "${BATCH_SIZE}" \
    --eval_batch_size "${BATCH_SIZE}" \
    --save \
    --output_dir "output/multi_turn/nheads${NUM_HEADS}nlayers${NUM_LAYERS}hdim${HIDDEN_SIZE}/lr${LR}bsize${BATCH_SIZE}/s${SEED}";
