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
    model = ToyModel(1, 2).to('cuda')
    x = torch.zeros(size=(1, 1, 1), dtype=torch.float16).to('cuda')
    gt = torch.zeros(size=(1, 1, 2), dtype=torch.float16).to('cuda')
    loss = nn.CrossEntropyLoss()

    optim = AdamW(model.parameters())

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        y = model(x)
        loss = loss(y, gt)

        loss.backward()

        optim.step()
