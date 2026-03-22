import os 
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse

parser = argparse.ArgumentParser()
 
parser.add_argument("--backend", help="", type=str)
parser.add_argument("--num_processes", help="", type=int)
parser.add_argument("--length", help="", type=int)

def setup(args, rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(args.backend, rank=rank, world_size=world_size)

def distributed_demo(args, rank, world_size):
    setup(rank, world_size)
    data = torch.randint(0, 10, (args.length,))
    print(f"rank {rank} data (before all-reduce): {data[:10]}")
    dist.all_reduce(args, data, async_op=False)
    print(f"rank {rank} data (after all-reduce): {data}")

def main(args):
    world_size = args.num_processes
    mp.spawn(fn=distributed_demo, args=(args, ), nprocs=world_size, join=True)



if __name__ == "__main__":
    main()