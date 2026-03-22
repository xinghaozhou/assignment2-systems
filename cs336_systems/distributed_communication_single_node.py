import os 
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import timeit

parser = argparse.ArgumentParser()
 
parser.add_argument("--backend", help="", type=str)
parser.add_argument("--num_processes", help="", type=int)
parser.add_argument("--length", help="", type=int)

args = parser.parse_args()


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group(args.backend, rank=rank, world_size=world_size)

def distributed_demo(rank, args, world_size):
    setup(rank, world_size)

    dist.barrier()

    if args.backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        data = torch.randint(0, 10, (args.length,), device=device)

    if args.backend == "nccl":
        torch.cuda.synchronize()
        start = timeit.default_timer()

    dist.all_reduce(data, async_op=False)

    if args.backend == "nccl":
        torch.cuda.synchronize()
        end = timeit.default_timer()

    if rank == 0: # only ask one device to print
        print(f"{args.backend} with {world_size} process for {args.length * 4 / (1024**2)}MB takes {((end - start)*1000):.2f} ms" )


def main(args):
    world_size = args.num_processes
    mp.spawn(fn=distributed_demo, args=(args, world_size), nprocs=world_size, join=True)



if __name__ == "__main__":
    main(args)