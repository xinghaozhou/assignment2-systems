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
class OverlapDDPBucketed(nn.Module):
    def __init__(self, module, bucket_size_mb: float):
        super().__init__()
        self.world_size = dist.get_world_size()
        self.module = module
        self.comm_time = 0
        self.handles = []
        self._reduced = set()
        self.bucket_size_mb = bucket_size_mb
        self.curr_size_mb = 0
        self.curr_params = []
        self.curr_grads = []

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

            self.curr_size_mb += grad.numel() * grad.element_size()
            self.curr_params.append(param) # Store bucket param
            self.curr_grads.append(grad) # Store bucket grads

            if self.bucket_size_mb < self.curr_size_mb: # Ready to comm (add list because otherwise, it passes the reference in)
                flatten_comm_grad = torch._utils._flatten_dense_tensors(list(self.curr_grads)) # Flatten bucket-stored grads 
                handle = dist.all_reduce(flatten_comm_grad, async_op=True) # Communication
                self.handles.append((handle, list(self.curr_params), list(self.curr_grads), flatten_comm_grad))

                # Clean 
                self.curr_size_mb = 0
                self.curr_params.clear()
                self.curr_grads.clear()

        return hook


    # after backward() before step(), deal with grad
    def finish_grad_synchronization(self):
        # handle just inform that the comm has been done
        for handle, params, grads, flatten_comm_grad in self.handles:
            start = timeit.default_timer()
            handle.wait()

            torch.cuda.synchronize()

            end = timeit.default_timer()

            unflatten_comm_tensor = torch._utils._unflatten_dense_tensors(flatten_comm_grad, grads)

            for p, grad in zip(params, unflatten_comm_tensor):
                if grad is not None:
                    grad.div_(self.world_size)
                    p.grad = grad


            self.comm_time += (end - start)

        self.handles.clear()
        self._reduced.clear()

        return 

    def train_batch_start(self, optim):
        optim.zero_grad()

        

    def forward(self, *args, **kwargs): # We don't know what passed in, so use *args **kwargs 
        return self.module(*args, **kwargs)        

    def get_comm_time(self):
        return self.comm_time



