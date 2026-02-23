import torch
import torch.nn as nn
from cs336_basics.optimizer import AdamW


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    model = ToyModel(10, 2)
    x = torch.zeros(size=(1, 1, 10), dtype=torch.float16)
    gt = torch.zeros(size=(1, 1, 2), dtype=torch.float16)

    optim = AdamW(model.parameters())

    with torch.autocast(device_type="cpu", dtype=torch.float16):
        y = model(x)
        loss = nn.CrossEntropy(y, gt)

        breakpoint()

        loss.backward()

        optim.step()
