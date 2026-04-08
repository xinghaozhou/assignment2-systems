import os 
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import timeit
import torch.nn as nn
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.model import BasicsTransformerLM
from cs336_systems.overlap_ddp_bucketed import OverlapDDPBucketed
from cs336_systems.optimizer_state_sharding import optimizer_state_sharding

parser = argparse.ArgumentParser()
 
parser.add_argument("--world_size", help="num of processes", type=int)
parser.add_argument("--num_iters", help="num of iterations", type=int)

parser.add_argument("--batch_size", help="batch size", type=int)
parser.add_argument("--vocab_size", help="vocab size", type=int)
parser.add_argument("--context_length", help="context length", type=int)
parser.add_argument("--d_model", help="dimension of model", type=int)
parser.add_argument("--num_layers", help="num of layer", type=int)
parser.add_argument("--num_heads", help="num of heads", type=int)
parser.add_argument("--d_ff", help="dimension of ffn", type=int)

parser.add_argument("--bucket_size_mb", help="size of the bucket", type=int)


args = parser.parse_args()

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    torch.cuda.set_device(rank)

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )


def ddp_demo(rank, world_size, args):
    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}")

    model = BasicsTransformerLM(args.vocab_size, args.context_length, args.d_model, args.num_layers, args.num_heads, args.d_ff, rope_theta=10000).to(device)

    optim = optimizer_state_sharding(model.parameters(), AdamW)
    overlap_ddp_bucketed = OverlapDDPBucketed(model, args.bucket_size_mb) # Wrap-up function, make it ready for communication

    x = torch.randint(0, 10, size=(args.batch_size, args.context_length)).to(device)
    gt = torch.randint(0, 10, size=(args.batch_size, args.context_length)).to(device)

    start = timeit.default_timer()

    torch.cuda.synchronize()

    for _ in range(args.num_iters):
        optim.zero_grad()
        y = overlap_ddp_bucketed(x)

        loss = cross_entropy(y, gt)

        loss.backward()

        overlap_ddp_bucketed.finish_grad_synchronization()
            
        optim.step()

    torch.cuda.synchronize()

    end = timeit.default_timer()

    if rank == 1:
        print(f"Total time for iterations is {((end - start) * 1000):.2f} ms. Total time for communication is {((overlap_ddp_bucketed.get_comm_time()) * 1000):.2f} ms.")

    dist.destroy_process_group()

def main(args):
    world_size = args.world_size
    mp.spawn(
        fn=ddp_demo,
        args=(world_size, args),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main(args)


   
    


