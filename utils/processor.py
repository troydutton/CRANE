from typing import Dict, Union

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import Tensor, nn, optim
from torch_geometric.data import Data
from tqdm import tqdm

import wandb
from model.gcn import GCN


def train_one_epoch(model: GCN, graph: Data, optimizer: optim.Optimizer, criterion: nn.CrossEntropyLoss, verbose: bool = True) -> None:
    model.train()
    optimizer.zero_grad()

    # Predict on training nodes
    logits = model(graph)[(mask := graph.train_mask)]
    credibility = graph.credibility[mask]

    # Compute the loss only on training nodes
    loss: Tensor = criterion(logits, torch.stack((1 - credibility, credibility), dim=1))
    
    if verbose:
        wandb.log({"Train": {"Loss": loss.item()}}, step=wandb.run.step + mask.sum().item())

    loss.backward()
    optimizer.step()

def evaluate(model: nn.Module, data, criterion: nn.CrossEntropyLoss, split: str = "val", credibility_threshold: float = 0.7) -> Dict[str, Union[float, int]]:
    model.eval()

    mask = data.train_mask if split == "train" else data.val_mask

    with torch.no_grad():
        # Predict on the nodes in the split
        logits = model(data)[mask]
        credibility = data.credibility[mask]
        labels = (credibility > credibility_threshold).long()

        probabilities = torch.softmax(logits, dim=-1)
        predicted_classes = torch.argmax(probabilities, dim=-1)

        loss = criterion(logits, torch.stack((1 - credibility, credibility), dim=1))

        accuracy = accuracy_score(labels, predicted_classes)
        precision = precision_score(labels, predicted_classes, zero_division=0.0)
        recall = recall_score(labels, predicted_classes)
        f1 = f1_score(labels, predicted_classes)
        auc = roc_auc_score(labels, probabilities[:, 1])

    return {
        "Loss": loss,
        "Metric": {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "AUC": auc
        }
    }

def train(model: GCN, data, optimizer: optim.Optimizer, epochs: int, criterion: nn.CrossEntropyLoss, credibility_threshold: float = 0.7) -> None:
    for _ in tqdm(range(epochs), desc="Training"):
        train_one_epoch(model, data, optimizer, criterion) 
        
        train_metrics = evaluate(model, data, criterion, split="train", credibility_threshold=credibility_threshold)

        val_metrics = evaluate(model, data, criterion, split="val", credibility_threshold=credibility_threshold)
        
        wandb.log({"Train": train_metrics, "Val": val_metrics}, step=wandb.run.step)
