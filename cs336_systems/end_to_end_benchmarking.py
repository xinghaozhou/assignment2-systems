import argparse
import numpy as np
import torch 
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.data import get_batch
from cs336_basics.optimizer import AdamW, get_cosine_lr
from cs336_basics.nn_utils import cross_entropy, clip_gradient

parser = argparse.ArgumentParser()
 
parser.add_argument("--device", help="device", default='mps', type=str)
parser.add_argument("--dtype", help="dtype", type=str)
parser.add_argument("--size", help="Model Size", type=str)

parser.add_argument("--d_model", help="d_model", type=int)
parser.add_argument("--d_ff", help="d_ff", type=int)
parser.add_argument("--num_layers", help="number of layers", type=int)
parser.add_argument("--num_heads", help="number of heads", type=int)


parser.add_argument("--batch_size", help="batch size", type=int)
parser.add_argument("--vocab_size", help="vocab size", type=int)
parser.add_argument("--context_length", help="length of context", type=int)
parser.add_argument("--rope_theta", help="theta in RoPE", type=float)

parser.add_argument("--warmup_steps", help="steps before start measuring time", type=int)
parser.add_argument("--pass_type", help="Type of passes when measuring time (e.g. forward)", type=str)
parser.add_argument("--test_steps", help="steps when start measuring time", type=int)

parser.add_argument("--dataset", help="random dataset", type=str)

args = parser.parse_args()

def main():
    model = BasicsTransformerLM(vocab_size=args.vocab_size, context_length=args.context_length, d_model=args.d_model, num_layers=args.num_layers, num_heads=args.num_heads, d_ff=args.d_ff, rope_theta=args.rope_theta).to(args.device)
    optim = AdamW(params=model.parameters(), lr=1.0, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    dataset = np.memmap(args.dataset, mode='r', dtype=np.int32)

    
    if args.pass_type == "forward":
        for w in range(args.warmup_steps):
            x, _ = get_batch(dataset=dataset, batch_size=args.batch_size, context_length=args.context_length, device=args.device)
            y = model(x)

        start = timeit.default_timer()
        for t in range(args.test_steps):
            x, _ = get_batch(dataset=dataset, batch_size=args.batch_size, context_length=args.context_length, device=args.device)
            y = model(x)
            torch.cuda.synchronize() # Make sure CPU and GPU aligned
            
        end = timeit.default_timer()

        print(f"Forward takes {(end - start):4f} ms")


    elif args.pass_type == "both":
        # Run w warm-up steps
        for w in range(args.warmup_steps):
            optim.zero_grad()

            # get data
            x, gt = get_batch(dataset=dataset, batch_size=args.batch_size, context_length=args.context_length, device=args.device)
            x = x.to(args.device)
            gt = gt.to(args.device)

            # learning rate (using a random value)
            lr = get_cosine_lr(w, max_learning_rate=1.0, min_learning_rate=1.0, warmup_iters=args.warmup_steps, cosine_cycle_iters=1)

            # optimzer update and step
            for group in optim.param_groups:
                group["lr"] = lr

            # get the result
            y = model(x)
        
            # get the loss
            loss = cross_entropy(y, gt)
            loss.backward()

            clip_gradient(model.parameters())
            optim.step()

        # Run t test steps
        start = timeit.default_timer()

        for t in range(args.test_steps):
            optim.zero_grad()

            # get data
            x, gt = get_batch(dataset=dataset, batch_size=args.batch_size, context_length=args.context_length, device=args.device)
            x = x.to(args.device)
            gt = gt.to(args.device)

            # learning rate (using a random value)
            lr = get_cosine_lr(w, max_learning_rate=1.0, min_learning_rate=1.0, warmup_iters=args.warmup_steps, cosine_cycle_iters=1)

            # optimzer update and step
            for group in optim.param_groups:
                group["lr"] = lr

            # get the result
            y = model(x)
        
            # get the loss
            loss = cross_entropy(y, gt)
            loss.backward()

            clip_gradient(model.parameters())
            optim.step()

        end = timeit.default_timer()

        print(f"Forward and Backward take {(end - start):4f} ms")


    else:
        raise ValueError("Argument of pass type does not exist")


if __name__ == "__main__":
    main()





