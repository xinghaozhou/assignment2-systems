#!/bin/bash

BATCH_SIZE=8

for D_MODEL in 16 32 64 128
do
  for SEQ_LEN in 256 1024 4096 8192 16384
  do
    echo "Running d_model=$D_MODEL seq_len=$SEQ_LEN"

    uv run python -m cs336_systems.attention \
    --device cuda \
    --dtype float32 \
    --head_embedding $D_MODEL\
    --seq_len $SEQ_LEN \
    --iteration \

  done
done


