import argparse
import torch
from einops import einsum, rearrange, einx
import torch.cuda.nvtx as nvtx
import timeit

from cs336_basics.model import scaled_dot_product_attention



parser = argparse.ArgumentParser()
 
parser.add_argument("--device", help="device", default='mps', type=str)
parser.add_argument("--dtype", help="dtype", type=str)

parser.add_argument("--head_embedding", help="head embedding", type=int)
parser.add_argument("--seq_len", help="sequence length for benchmark", type=int)
parser.add_argument("--iteration", help="num of iterations for testing", type=int)

args = parser.parse_args()

def main():
    B = 8
    warmup_iters = 5
    iters = 100

    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "float32":
        dtype = torch.float32
    else:
        raise ValueError("Invalid dtype")

    Q = torch.randn(B, args.seq_len, args.head_embedding, device=args.device, dtype=dtype, requires_grad=True)
    K = torch.randn(B, args.seq_len, args.head_embedding, device=args.device, dtype=dtype, requires_grad=True)
    V = torch.randn(B, args.seq_len, args.head_embedding, device=args.device, dtype=dtype, requires_grad=True)

    mask = torch.triu(
        torch.ones(args.seq_len, args.seq_len, device=args.device),
        diagonal=1
    )

    mask = mask.masked_fill(mask == 1, float("-inf"))
    mask = mask.to(dtype)

    for _ in range(warmup_iters):
        for t in (Q, K, V):
            t.grad = None
        attn_scores = scaled_dot_product_attention(Q, K, V, mask, device=args.device, dtype=dtype)
        loss = attn_scores.sum()
        loss.backward()
        

    torch.cuda.synchronize()
    start = timeit.default_timer()

    with torch.no_grad():
        for _ in range(iters):
            attn_scores = scaled_dot_product_attention(Q, K, V, mask)
            torch.cuda.synchronize()
            
    end = timeit.default_timer()

    print(f"Combination: {args.seq_len} {args.head_embedding}, Total Time of Forward takes {(end - start):.2f}, takes {((end - start) / iters):.2f} per forward.")

    torch.cuda.memory._record_memory_history(max_entries=1000000)
    attn_scores = scaled_dot_product_attention(Q, K, V, mask)
             
    torch.cuda.memory._dump_snapshot(f"{args.seq_len}_{args.head_embedding}_memory_snapshot.pickle")
                
    torch.cuda.memory._record_memory_history(enabled=None)

    torch.cuda.synchronize()
    start = timeit.default_timer()

    for _ in range(iters):
        for t in (Q, K, V):
            t.grad = None
        attn_scores = scaled_dot_product_attention(Q, K, V, mask)
        loss = attn_scores.sum()
        loss.backward()
        torch.cuda.synchronize()

    end = timeit.default_timer()
    print(f"Combination: {args.seq_len} {args.head_embedding}, Total Time of Backward takes {(end - start):.2f}, takes {((end - start) / iters):.2f} per forward.")





    return attn_scores


if __name__ == "__main__":
    main()

    


       







