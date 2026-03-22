
for num_processes in 2 4 6
do
    for length in 250000 2500000 25000000 250000000
    do
        for backend in gloo NCCL
        do
            uv run python cs336_systems.distributed_communication_single_node \
                --backend $backend \
                --num_processes $num_processes \
                --length $length \
                
        done
    done
done