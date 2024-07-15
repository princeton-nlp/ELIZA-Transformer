#!/bin/bash

DATASET="alpha0.1"
NUM_EPOCHS=100
BATCH_SIZE=64
LR=0.0001
NUM_HEADS=12
NUM_LAYERS=8
HIDDEN_SIZE=768
EXP_NAME="single_turn"
SEED=0

python src/run.py \
    --data_dir "data/single_turn/${DATASET}" \
    --num_layers "${NUM_LAYERS}" \
    --num_heads "${NUM_HEADS}" \
    --hidden_size "${HIDDEN_SIZE}" \
    --lr "${LR}" \
    --num_epochs "${NUM_EPOCHS}" \
    --seed "${SEED}" \
    --train_batch_size "${BATCH_SIZE}" \
    --eval_batch_size "${BATCH_SIZE}" \
    --eval_on "data/single_turn/alpha0.01" "data/single_turn/alpha0.1" "data/single_turn/alpha1.0" "data/single_turn/alpha100.0" \
    --save \
    --output_dir "output/${EXP_NAME}/${DATASET}/nheads${NUM_HEADS}nlayers${NUM_LAYERS}hdim${HIDDEN_SIZE}/lr${LR}bsize${BATCH_SIZE}/s${SEED}";
