#!/bin/bash

python src/generate_data.py \
    --output_dir "data/multi_turn" \
    --num_templates 32 \
    --max_transformations_per_template 5 \
    --null_template \
    --null_template_cycle_version 1 \
    --memory_queue \
    --max_queue_size 4 \
    --memory_ratio 8 \
    --max_template_len 10 \
    --max_ngram_len 3 \
    --min_num_wildcards 2 \
    --max_num_wildcards 4 \
    --template_prefix_len 2 \
    --min_copy_len 0 \
    --max_copy_len 10 \
    --num_conversations 140000 \
    --max_turns 100 \
    --max_conversation_len 512 \
    --concentration 0.03125 \
    --unigram_concentration "-1" \
    --seed 0;
