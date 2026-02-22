uv run nsys profile -o result python benchmark.py \
    --device cuda \
    --dtype float16 \
    --d_model 2560 \
    --d_ff 10240 \
    --num_layers 32 \
    --num_heads 32 \
    --warmup_steps 5 \
    --pass_type both \
    --test_steps 10 \
