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

# for bucket_size in 1 10 100 1000
# do 
#     uv run python -m cs336_systems.benchmark_overlap_ddp_bucketed \
#         --world_size 2 \
#         --num_iters 10\
#         --batch_size 4 \
#         --vocab_size 10000\
#         --context_length 128\
#         --d_model 768\
#         --num_layers 12\
#         --num_heads 12\
#         --d_ff 3072\
#         --bucket_size_mb $bucket_size \

# done


for bucket_size in 1 10 100 1000
do 
    uv run python -m cs336_systems.benchmark_overlap_ddp_bucketed \
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



