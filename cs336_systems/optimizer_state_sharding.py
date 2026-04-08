import os 
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import timeit
import os
import torch.nn as nn
from typing import Type, Any
from torch.optim import Optimizer
from cs336_systems.ToyModel import ToyModel
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy


class optimizer_state_sharding(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.params = list(params) # To keep all the params

        defaults = {}
        super().__init__(self.params, defaults)
        # constructor will call add_param_group 

        # now, get param->rank dict
        self.param_to_rank = self.build_param_rank_mapping()

        # Build the local param list
        local_param = []
        for group in self.param_groups:
            for p in group["params"]:
                if self.param_to_rank[p] != self.rank:
                    continue
                else:
                    local_param.append(p)

        self.optimizer = optimizer_cls(local_param, **kwargs)
        
    def step(self, *args, **kwargs):
        self.optimizer.step(*args, **kwargs)
        # After step, write the param in current rank back together
        for p in self.params:
            owner = self.param_to_rank[p]
            dist.broadcast(p.data, src=owner)

    def build_param_rank_mapping(self):
        # Here, in param_group, the structure is like: {str: corresponding values}
        params = []

        # Assign each param to each rank based on their size
        for group in self.param_groups:
            for p in group['params']:
                params.append(p)

        params = sorted(params, key=lambda p: p.numel(), reverse=True)

        param_to_rank = {}
        for i, p in enumerate(params):
            rank_idx = i % self.world_size
            param_to_rank[p] = rank_idx

        return param_to_rank

        

