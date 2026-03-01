<<<<<<< HEAD
uv run nsys profile -o result python -m cs336_systems.benchmark \
    --device cuda \
    --dtype float32 \
    --size 2.7B \
    --use_bf16 True \
    --memory_record True \
    --d_model 2560 \
    --d_ff 10240 \
    --num_layers 32 \
    --num_heads 32 \
    --warmup_steps 5 \
    --pass_type both \
    --test_steps 10 \
=======
uv run nsys profile -o result python -m cs336_systems.benchmark \
    --device cuda \
    --dtype float32 \
    --size 2.7B \
    --use_memory_record \
    --d_model 2560 \
    --d_ff 10240 \
    --num_layers 32 \
    --num_heads 32 \
    --warmup_steps 0 \
    --pass_type both \
    --test_steps 10 \
>>>>>>> b6b1a4d (extra repeated file remove, ready for FlashAttention)
