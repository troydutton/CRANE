from torch import nn, optim
from torch_geometric.loader import DataLoader

from model.gcn import GCN


def train_one_epoch(model: GCN, dataloader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module) -> None:
    model.train()

    for batch in dataloader:
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)  # Assuming you have labels in `data.y`
        loss.backward()
        optimizer.step()

def train(model: GCN, dataloader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, epochs: int) -> None:
    for epoch in range(epochs):
        train_one_epoch(model, dataloader, optimizer, criterion)