from typing import Dict, Union

from torch import nn, optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import wandb
from model.gcn import GCN


def train_one_epoch(model: GCN, dataloader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module) -> None:
    model.train()

    for batch in dataloader:
        optimizer.zero_grad()

        # Forward pass
        predictions = model(batch)

        # Backward pass
        loss = criterion(predictions, batch.credibility)

        wandb.log({"Train": {"Loss": loss.item()}}, step=wandb.run.step + batch[0].num_nodes)

        loss.backward()
        optimizer.step()


def train(model: GCN, dataloader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, epochs: int) -> None:
    for _ in tqdm(range(epochs), desc="Training"):
        train_one_epoch(model, dataloader, optimizer, criterion)