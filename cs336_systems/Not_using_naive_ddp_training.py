import os 
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import timeit
import torch.nn as nn
from cs336_systems.ToyModel import ToyModel
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy

parser = argparse.ArgumentParser()
 
parser.add_argument("--num_processes", help="num of processes", type=int)
parser.add_argument("--batch_size", help="num of batch", type=int)
parser.add_argument("--seq_len", help="length of data", type=int)
parser.add_argument("--iteration", help="num of iters", type=int)
parser.add_argument("--benchmark", action="store_true")

args = parser.parse_args()


class DDPIndividualParameters(nn.Module):

    def __init__(self, num_processes, batch_size, seq_len, iteration):
        super().__init__()
        self.num_processes = num_processes
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.iteration = iteration

    def setup(self, rank, world_size):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"

        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
        )
 
    def ddp_demo(self, rank, iteration, world_size, x, gt):
        self.setup(rank, world_size)
        device = torch.device(f"cuda:{rank}")

        model = ToyModel(x.size(-1), gt.size(-1)).to(device)
        optim = AdamW(model.parameters())

        torch.cuda.synchronize()

        for _ in range(iteration):
            optim.zero_grad()

            local_x = x.chunk(world_size)[rank].to(device)
            local_gt = gt.chunk(world_size)[rank].to(device)

            local_y = model(local_x)
            local_loss = cross_entropy(local_y, local_gt)

            local_loss.backward()
            # Need to manually update for each rank
            for params in model.parameters():
                dist.all_reduce(params.grad)
                params.grad /= world_size

                optim.step()

        dist.destroy_process_group()

    def start(self):
        world_size = self.num_processes
        batch = self.batch_size
        seq_len = self.seq_len
        iteration = self.iteration
        x = torch.randint(0, 10, (batch, seq_len), dtype=torch.float32)
        gt = torch.randint(0, 10, (batch,), dtype=torch.int64)

        mp.spawn(fn=self.ddp_demo, args=(iteration, world_size, x, gt), nprocs=world_size, join=True)

if __name__ == "__main__":
    ddp = DDPIndividualParameters(args.num_processes, args.batch_size, args.seq_len, args.iteration)
    ddp.start()