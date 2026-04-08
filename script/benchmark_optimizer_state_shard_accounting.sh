#!/bin/bash

for bucket_size in 1 10 100 1000
do 
    uv run python -m cs336_systems.benchmark_optimizer_state_sharding \
        --world_size 2 \
        --num_iters 10\
        --batch_size 4 \
        --vocab_size 10000\
        --context_length 128\
        --d_model 1280\
        --num_layers 36\
        --num_heads 20\
        --d_ff 5120\
        --bucket_size_mb $bucket_size \

done
