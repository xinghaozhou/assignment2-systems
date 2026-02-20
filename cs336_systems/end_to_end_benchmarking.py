import argparse
import numpy as np
import torch 
import timeit
import statistics
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy

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

def main():
    batch_size = 4
    context_length = 256
    vocab_size = 10000
    rope_theta = rope_theta

    model = BasicsTransformerLM(vocab_size=vocab_size, context_length=context_length, d_model=args.d_model, num_layers=args.num_layers, num_heads=args.num_heads, d_ff=args.d_ff, rope_theta=rope_theta).to(args.device, args.dtype)
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
                y = model(x)
                torch.cuda.synchronize() # Make sure CPU and GPU aligned
                end = timeit.default_timer()
                duration = end - start 
                time_list.append(duration)

        print(f"Forward Only, Mean: {statistics.mean(time_list)}, Std: {statistics.stdev(time_list)}")

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

            # get the result
            y = model(x)
        
            # get the loss
            loss = cross_entropy(y, gt)
            loss.backward()

            optim.step()

            torch.cuda.synchronize() # Make sure CPU and GPU aligned

            end = timeit.default_timer()

            duration = end - start 
            time_list.append(duration)

        print(f"Both, Mean: {statistics.mean(time_list)}, Std: {statistics.stdev(time_list)}")

    else:
        raise ValueError("Argument of pass type does not exist")


if __name__ == "__main__":
    main()





