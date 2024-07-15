#!/bin/bash

for alpha in 0.01 0.1 1.0 100.0; do
    python src/generate_data.py \
        --output_dir "data/single_turn/alpha${alpha}" \
        --num_templates 15 \
        --null_template \
        --max_transformations_per_template 1 \
        --max_template_len 10 \
        --max_ngram_len 1 \
        --min_num_wildcards 2 \
        --max_num_wildcards 2 \
        --template_prefix_len 1 \
        --min_copies 2 \
        --max_copies 2 \
        --min_copy_len 0 \
        --max_copy_len 20 \
        --num_conversations 65536 \
        --max_turns 1 \
        --max_conversation_len 512 \
        --concentration 1.0 \
        --unigram_concentration "${alpha}" \
        --seed 0;
done;