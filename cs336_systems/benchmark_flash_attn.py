import argparse
import numpy as np
import torch 
import timeit
import statistics

from contextlib import nullcontext
from cs336_basics.model import BasicsTransformerLM

from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
from cs336_systems.FlashAttention_pytorch import FlashAttentionPytorch
from cs336_systems.FlashAttention_triton import FlashAttentionTriton

import torch.cuda.nvtx as nvtx
import triton

parser = argparse.ArgumentParser()
 
parser.add_argument("--device", help="device", default='mps', type=str)
parser.add_argument("--dtype", help="dtype", type=str)

# use this as bool args
parser.add_argument("--use_triton", action="store_true")
parser.add_argument("--test_type", help="forward, backward, or end_to_end", type=str)

# experiment setup
parser.add_argument("--batch", help="batch size for benchmark", type=int)
parser.add_argument("--seq_len", help="context length for benchmark", type=int)
parser.add_argument("--d_model", help="dimension size", type=int)

args = parser.parse_args()

def bench_forward(
    fn, Q, K, V, mask
):
    return triton.testing.do_bench(lambda: fn.apply(Q, K, V, mask=mask))

def bench_backward(
    fn, Q, K, V, mask
):
    out = fn.apply(Q, K, V, mask)
    grad = torch.randn_like(out)

    def run():
        Q.grad = None
        K.grad = None
        V.grad = None
        out.backward(grad, retain_graph=True)

    return triton.testing.do_bench(run)

def bench_end_to_end(
    fn, Q, K, V, mask
):
    def run():
        Q.grad = None
        K.grad = None
        V.grad = None
        o = fn.apply(Q, K, V, mask=mask)
        loss = o.sum()
        loss.backward()
    return triton.testing.do_bench(run)
   


def main():
    batch_size = args.batch
    context_length = args.seq_len
    d_model = args.d_model
    dtype = args.dtype
    device= args.device

    if dtype == "bf16":
        Q = torch.randn(batch_size, context_length, d_model, device=device, dtype=torch.bfloat16, requires_grad=True)
        K = torch.randn(batch_size, context_length, d_model, device=device, dtype=torch.bfloat16, requires_grad=True)
        V = torch.randn(batch_size, context_length, d_model, device=device, dtype=torch.bfloat16, requires_grad=True)

    elif dtype == "fp32":
        Q = torch.randn(batch_size, context_length, d_model, device=device, dtype=torch.float32, requires_grad=True)
        K = torch.randn(batch_size, context_length, d_model, device=device, dtype=torch.float32, requires_grad=True)
        V = torch.randn(batch_size, context_length, d_model, device=device, dtype=torch.float32, requires_grad=True)
  
    else:
       raise TypeError(f"Invalid dtype {dtype}")
    
    if not args.use_triton: 
        if args.test_type == "forward":
            bench_forward(FlashAttentionPytorch, Q, K, V, mask=True)
        elif args.test_type == "backward":
            bench_backward(FlashAttentionPytorch, Q, K, V, mask=True)
        elif args.test_type == "end_to_end":
            bench_end_to_end(FlashAttentionPytorch, Q, K, V, mask=True)
    else:
        if args.test_type == "forward":
            bench_forward(FlashAttentionTriton, Q, K, V, mask=True)
        elif args.test_type == "backward":
            bench_backward(FlashAttentionTriton, Q, K, V, mask=True)
        elif args.test_type == "end_to_end":
            bench_end_to_end(FlashAttentionTriton, Q, K, V, mask=True)


        
    