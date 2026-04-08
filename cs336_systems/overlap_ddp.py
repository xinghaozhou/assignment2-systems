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
class OverlapDDP(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.world_size = dist.get_world_size()
        self.module = module
        self.comm_time = 0
        self.handles = []
        self._reduced = set()

        # To broadcast all initial weights 
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)

        # To register all the parameters so that when loss.backward(), it gets updated
        for p in self.module.parameters():
            if p.requires_grad:
                # Because a grad can be use for multiple time, wait this to be accumulated
                p.register_post_accumulate_grad_hook(self._make_hook(p))


    # make hook, function to build comm when grad is ready
    # So it won't block backward()
    def _make_hook(self, p):
        def hook(param):
            # Make sure grad is contiguous

            if param in self._reduced:
                return

            grad = param.grad

            if grad is None:
                return

            self._reduced.add(param)

            if not grad.is_contiguous():
                grad = grad.contiguous()

            handle = dist.all_reduce(grad, async_op=True)

            self.handles.append((handle, param, grad))

        return hook


    # after backward() before step(), deal with grad
    def finish_grad_synchronization(self):
        # handle just inform that the comm has been done
        for handle, p, grad in self.handles:
            start = timeit.default_timer()
        
            handle.wait()
            torch.cuda.synchronize()

            end = timeit.default_timer()

            self.comm_time += (end - start)

            grad.div_(self.world_size)
            p.grad = grad

        self.handles.clear()
        self._reduced.clear()

        return 
        

    def forward(self, *args, **kwargs): # We don't know what passed in, so use *args **kwargs 
        return self.module(*args, **kwargs)        

    def get_comm_time(self):
        return self.comm_time



