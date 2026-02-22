import argparse
import numpy as np
import torch 
import timeit
import statistics
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy

import torch.cuda.nvtx as nvtx

parser = argparse.ArgumentParser()
 
parser.add_argument("--device", help="device", default='mps', type=str)
parser.add_argument("--dtype", help="dtype", type=str)
parser.add_argument("--size", help="Model Size", type=str)

parser.add_argument("--d_model", help="d_model", type=int)
parser.add_argument("--d_ff", help="d_ff", type=int)
parser.add_argument("--num_layers", help="number of layers", type=int)
parser.add_argument("--num_heads", help="number of heads", type=int)

parser.add_argument("--warmup_steps", help="steps before start measuring time", type=int)
parser.add_argument("--pass_type", help="Type of passes when measuring time (e.g. forward)", type=str)
parser.add_argument("--test_steps", help="steps when start measuring time", type=int)

args = parser.parse_args()

import math
from einops import rearrange, einsum
from cs336_basics.nn_utils import softmax


@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
   Q, K, V, mask
  # Q, K, V, mask
):
    d_k = K.shape[-1]

    with nvtx.range("computing attention scores"):
      attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

      if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("computing softmax"):
      attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension

    with nvtx.range("final matmul"):
       output = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")

    return output
   
   
cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention



def main():
    batch_size = 4
    context_length = 128
    vocab_size = 10000
    rope_theta = 10000

    if args.size == "small":
      print(f"size = {args.size}")
      d_model=768
      d_ff=3072
      num_layers=12
      num_heads=12
    elif args.size == "medium":
      print(f"size = {args.size}")
      d_model=1024
      d_ff=4096
      num_layers=24
      num_heads=16
    elif args.size == "large":
      print(f"size = {args.size}")
      d_model=1290
      d_ff=5120
      num_layers=36
      num_heads=20
    elif args.size == "xl":
      print(f"size = {args.size}")
      d_model=1600
      d_ff=6400
      num_layers=48
      num_heads=25
    elif args.size == "2.7B":
      print(f"size = {args.size}")
      d_model=2560
      d_ff=6400
      num_layers=32
      num_heads=32
    else:
      d_model=args.d_model
      d_ff=args.d_ff
      num_layers=args.num_layers
      num_heads=args.num_heads
    

    if args.dtype == "float16":
      model = BasicsTransformerLM(vocab_size=vocab_size, context_length=context_length, d_model=d_model, num_layers=num_layers, num_heads=num_heads, d_ff=d_ff, rope_theta=rope_theta).to(args.device, torch.float16)
    elif args.dtype == "float32":
      model = BasicsTransformerLM(vocab_size=vocab_size, context_length=context_length, d_model=d_model, num_layers=num_layers, num_heads=num_heads, d_ff=d_ff, rope_theta=rope_theta).to(args.device, torch.float32)

    optim = AdamW(params=model.parameters(), lr=1.0, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    time_list = []

    # get data
    x = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, context_length),
        device=args.device
    )

    gt = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, context_length),
        device=args.device
    )
    

    
    if args.pass_type == "forward":
        with torch.no_grad():
            for _ in range(args.warmup_steps):
                y = model(x)

            for _ in range(args.test_steps):
                start = timeit.default_timer()
                with nvtx.range("forward"):
                  y = model(x)
                torch.cuda.synchronize() # Make sure CPU and GPU aligned
                end = timeit.default_timer()
                duration = end - start 
                time_list.append(duration)

        print(f"Forward Only, Mean: {statistics.mean(time_list):.2f}, Std: {statistics.stdev(time_list):.2f}")

    elif args.pass_type == "both":
        # Run w warm-up steps
        for _ in range(args.warmup_steps):
            optim.zero_grad()

            # get the result
            y = model(x)
        
            # get the loss
            loss = cross_entropy(y, gt)
            loss.backward()

            # update the optimizer
            optim.step()

        for _ in range(args.test_steps):
            start = timeit.default_timer()
            optim.zero_grad()

            with nvtx.range("forward"):
              # get the result
              y = model(x)
              # get the loss
              loss = cross_entropy(y, gt)
        
            with nvtx.range("backward"):
              loss.backward()

            with nvtx.range("optimizer"):
              optim.step()

            torch.cuda.synchronize() # Make sure CPU and GPU aligned

            end = timeit.default_timer()

            duration = end - start 
            time_list.append(duration)

        print(f"Both, Mean: {statistics.mean(time_list):.2f}, Std: {statistics.stdev(time_list):.2f}")

    else:
        raise ValueError("Argument of pass type does not exist")


if __name__ == "__main__":
    main()



