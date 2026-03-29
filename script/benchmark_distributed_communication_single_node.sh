
for num_processes in 2 4 6
do
    for length in 262144 2621440 26214400 262144000
    do
        for backend in nccl
        do
            uv run python -m cs336_systems.distributed_communication_single_node \
                --backend $backend \
                --num_processes $num_processes \
                --length $length 

        done
    done
done