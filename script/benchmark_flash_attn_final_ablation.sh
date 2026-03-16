BATCH_SIZE=1

for D_MODEL in 16 32 64 128
do
    for SEQ_LEN in 128 256 512 1024 2048 4096 8192 16382 32768 65536
    do
        for type in bf16 fp32
        do 
            for test_type in forward backward end_to_end
            do
                echo "Using Triton!"
                echo "Running d_model=$D_MODEL seq_len=$SEQ_LEN type=$type test_type=$test_type"
                uv run python -m cs336_systems.benchmark_flash_attn \
                    --device cuda \
                    --dtype $type  \
                    --use_triton \
                    --test_type $test_type \
                    --batch 1 \
                    --d_model $D_MODEL \
                    --seq_len $SEQ_LEN \ 
            done
        done
    done
done

for D_MODEL in 16 32 64 128
do
    for SEQ_LEN in 128 256 512 1024 2048 4096 8192 16382 32768 65536
    do
        for type in bf16 fp32
        do 
            for test_type in forward backward end_to_end
            do
                echo "Not using Triton!"
                echo "Running d_model=$D_MODEL seq_len=$SEQ_LEN type=$type test_type=$test_type"
                uv run python -m cs336_systems.benchmark_flash_attn \
                    --device cuda \
                    --dtype $type  \
                    --test_type $test_type \
                    --batch 1 \
                    --d_model $D_MODEL \
                    --seq_len $SEQ_LEN \ 
            done
        done
    done
done
