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
# - register the hook (graident sync)
# - forward
class NaiveDDP(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.world_size = dist.get_world_size()
        self.module = module
        self.comm_time = 0
        self.comm_tensor = []

        # To broadcast all initial weights 
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)
        
        # To register all the parameters so that when loss.backward(), it gets updated
        for p in self.module.parameters():
            if p.requires_grad:
                p.register_hook(self._make_hook())

            
    # make hook, function to all_reduce the grad
    def _make_hook(self):
        def hook(grad):
            # Make sure grad is contiguous
            start = timeit.default_timer()

            grad = grad.contiguous()

            torch.cuda.synchronize()
            dist.all_reduce(grad, async_op=False)

            torch.cuda.synchronize()
            end = timeit.default_timer()
            grad = grad / self.world_size

            self.comm_time += (end - start)
            return grad
        return hook


    def forward(self, *args, **kwargs): # We don't know what passed in, so use *args **kwargs 
        return self.module(*args, **kwargs)        

    def get_comm_time(self):
        return self.comm_time



