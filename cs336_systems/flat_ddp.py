import os 
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import timeit
import os
import torch.nn as nn
from cs336_systems.ToyModel import ToyModel
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy


# A model Wrapper to 
# - manage the parameters
# - forward
class FlatDDP(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.world_size = dist.get_world_size()
        self.module = module
        self.comm_time = 0

        # To broadcast all initial weights 
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)
        
    def finish_grad_synchronization(self):
        comm_params = []
        comm_grads = []
        for p in self.module.parameters():
            if p.requires_grad and p.grad is not None:

                p.grad = p.grad.contiguous() if not p.grad.is_contiguous() else p.grad# Make sure it is contiguous

                comm_params.append(p)
                comm_grads.append(p.grad)
            
        flatten_comm_tensor = torch._utils._flatten_dense_tensors(comm_grads)

        torch.cuda.synchronize()
        start = timeit.default_timer()

        dist.all_reduce(flatten_comm_tensor)

        torch.cuda.synchronize()
        end = timeit.default_timer()

        flatten_comm_tensor = flatten_comm_tensor / self.world_size
        unflatten_comm_tensor = torch._utils._unflatten_dense_tensors(flatten_comm_tensor, comm_grads)

        for p, grad in zip(comm_params, unflatten_comm_tensor):
            if p.requires_grad and p.grad is not None:
                p.grad.copy_(grad)

        self.comm_time += (end - start)

        return 
        

    def forward(self, *args, **kwargs): # We don't know what passed in, so use *args **kwargs 
        return self.module(*args, **kwargs)        

    def get_comm_time(self):
        return self.comm_time

