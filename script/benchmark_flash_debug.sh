

$D_MODEL = 32
for test_type in backward end_to_end
do
    for SEQ_LEN in 8192 16384 
    do
        for dtype in bf16 fp32
        do
            uv run python -m cs336_systems.benchmark_flash_attn \
                --device cuda \
                --dtype bf16  \
                --use_triton \
                --test_type $test_type \
                --batch 1 \
                --d_model $D_MODEL \
                --seq_len $SEQ_LEN \

            uv run python -m cs336_systems.benchmark_flash_attn \
                --device cuda \
                --dtype bf16  \
                --test_type $test_type \
                --batch 1 \
                --d_model $D_MODEL \
                --seq_len $SEQ_LEN \

    done
done




