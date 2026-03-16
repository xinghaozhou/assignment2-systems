uv run python -m cs336_systems.benchmark_flash_attn \
    --device cuda \
    --dtype bf16  \
    --use_triton True \
    --test_type backward \
    --batch 1 \
    --seq_len 128 \ 
    --d_model 16 \

