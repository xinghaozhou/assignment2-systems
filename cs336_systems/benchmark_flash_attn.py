import argparse
import numpy as np
import torch 
import timeit
import statistics
import math
import einx
from einops import einsum


from contextlib import nullcontext
from cs336_basics.model import BasicsTransformerLM

from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import softmax
from cs336_systems.FlashAttention_triton import FlashAttentionTriton

import torch.cuda.nvtx as nvtx
import triton

parser = argparse.ArgumentParser()
 
parser.add_argument("--device", help="device", default='cuda', type=str)
parser.add_argument("--dtype", help="dtype", type=str)

# use this as bool args
parser.add_argument("--use_triton", action="store_true")
parser.add_argument("--test_type", help="forward, backward, or end_to_end", type=str)

# experiment setup
parser.add_argument("--batch", help="batch size for benchmark", type=int)
parser.add_argument("--seq_len", help="context length for benchmark", type=int)
parser.add_argument("--d_model", help="dimension size", type=int)

args = parser.parse_args()

def call_fn(fn, Q, K, V, mask):
    # Triton: autograd.Function
    if hasattr(fn, "apply"):
        return fn.apply(Q, K, V, mask)
    else:
        return fn(Q, K, V, mask)

def bench_forward(
    fn, Q, K, V, mask
):
    return triton.testing.do_bench(lambda: call_fn(fn, Q, K, V, mask), warmup=10)

def bench_backward(fn, Q, K, V, mask):
    Q_ = Q.detach().clone().requires_grad_(True)
    K_ = K.detach().clone().requires_grad_(True)
    V_ = V.detach().clone().requires_grad_(True)

    out = call_fn(fn, Q_, K_, V_, mask)
    grad = torch.randn_like(out)

    def run():
        Q_.grad = None
        K_.grad = None
        V_.grad = None

        out.backward(grad, retain_graph=True)

    return triton.testing.do_bench(run, warmup=10)

def bench_end_to_end(fn, Q, K, V, mask):
    def run():
        Q_ = Q.detach().clone().requires_grad_(True)
        K_ = K.detach().clone().requires_grad_(True)
        V_ = V.detach().clone().requires_grad_(True)

        out = call_fn(fn, Q_, K_, V_, mask)
        grad = torch.ones_like(out)

        out.backward(grad)

    return triton.testing.do_bench(run, warmup=10)
   
def regular_pytorch_implementation(Q, K, V, mask):        
    d_k = Q.shape[-1]
    attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    if mask:
        mask_tensor = torch.triu(
            torch.ones((attention_scores.size(-2), attention_scores.size(-1)), device=attention_scores.device),
            diagonal=1
        ) == 0   

        attention_scores = attention_scores.masked_fill(~mask_tensor, float("-inf")) # Do masking first, Put -inf for those ~mask (True mask)

    attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension

    output = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
    return output



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
            latency = bench_forward(regular_pytorch_implementation, Q, K, V, mask=True)
        elif args.test_type == "backward":
            latency = bench_backward(regular_pytorch_implementation, Q, K, V, mask=True)
        elif args.test_type == "end_to_end":
            latency = bench_end_to_end(regular_pytorch_implementation, Q, K, V, mask=True)
        else:
            raise TypeError(f"Invavlid test_type {args.test_type}.")
    else:
        if args.test_type == "forward":
            latency = bench_forward(FlashAttentionTriton, Q, K, V, mask=True)
        elif args.test_type == "backward":
            latency = bench_backward(FlashAttentionTriton, Q, K, V, mask=True)
        elif args.test_type == "end_to_end":
            latency = bench_end_to_end(FlashAttentionTriton, Q, K, V, mask=True)
        else:
            raise TypeError(f"Invavlid test_type {args.test_type}.")

    print(f"{args.test_type} Latency is {(latency):.2f} ms")

        
if __name__ == "__main__":
    main()