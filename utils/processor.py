from typing import Dict, Union

import torch
from torch import nn, optim
from tqdm import tqdm

import wandb
from model.gcn import GCN


def train_one_epoch(model: GCN, data, optimizer: optim.Optimizer, criterion: nn.Module) -> None:
    model.train()
    optimizer.zero_grad()

    # Select training nodes
    train_mask = data.train_mask
    train_predictions = model(data)[train_mask]
    train_labels = data.credibility[train_mask]

    # Compute the loss only on training nodes
    loss = criterion(train_predictions, train_labels)

    # Log training loss
    wandb.log({"Train": {"Loss": loss.item()}}, step=wandb.run.step + train_mask.sum().item())

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

def evaluate(model: nn.Module, data, criterion: nn.Module, split: str = "val") -> Dict[str, Union[float, int]]:
    model.eval()
    total_loss = 0
    accuracy = 0
    num_nodes = 0

    # Choose the appropriate mask based on the split
    mask = data.train_mask if split == "train" else data.val_mask

    with torch.no_grad():
        # Select nodes based on the chosen mask
        predictions = model(data)[mask]
        labels = data.credibility[mask]

        # Compute the loss
        loss = criterion(predictions, labels)
        total_loss += loss.item() * mask.sum().item()

        # Calculate accuracy
        predicted_classes = predictions.argmax(dim=1)
        accuracy += (predicted_classes == labels).sum().item()
        num_nodes += mask.sum().item()
    
    metrics = {
        "Loss": total_loss / num_nodes,
        "Metric": {
            "Accuracy": accuracy / num_nodes
        }
    }
    
    return metrics

def train(model: GCN, data, optimizer: optim.Optimizer, criterion: nn.Module, epochs: int) -> None:
    for _ in tqdm(range(epochs), desc="Training"):
        train_one_epoch(model, data, optimizer, criterion)
        
        train_metrics = evaluate(model, data, criterion, split="train")

        val_metrics = evaluate(model, data, criterion, split="val")
        
        # Log metrics
        wandb.log({"Train": train_metrics, "Val": val_metrics}, step=wandb.run.step)
