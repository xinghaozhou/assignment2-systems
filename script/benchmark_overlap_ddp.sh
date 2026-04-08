#!/bin/bash

# uv run python -m cs336_systems.benchmark_flat_ddp \
#     --world_size 2 \
#     --num_iters 10\
#     --batch_size 4\
#     --vocab_size 10000\
#     --context_length 128\
#     --d_model 1600\
#     --num_layers 48\
#     --num_heads 25\
#     --d_ff 6400\

uv run python -m cs336_systems.benchmark_overlap_ddp \
    --world_size 2 \
    --num_iters 10\
    --batch_size 4 \
    --vocab_size 10000\
    --context_length 128\
    --d_model 1280\
    --num_layers 36\
    --num_heads 20\
    --d_ff 5120\




