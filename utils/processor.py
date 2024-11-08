from typing import Dict, Union

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import Tensor, nn, optim
from torch.nn.functional import cross_entropy
from torch_geometric.data import Data
from tqdm import tqdm

import wandb
from model.gcn import GCN


def train_one_epoch(model: GCN, graph: Data, optimizer: optim.Optimizer) -> None:
    model.train()
    optimizer.zero_grad()

    # Predict on training nodes
    logits = model(graph)[(mask := graph.train_mask)]
    labels = graph.credibility[mask]

    # Compute the loss only on training nodes
    loss = cross_entropy(logits, labels)

    wandb.log({"Train": {"Loss": loss.item()}}, step=wandb.run.step + mask.sum().item())

    loss.backward()
    optimizer.step()

def evaluate(model: nn.Module, data, split: str = "val") -> Dict[str, Union[float, int]]:
    model.eval()

    mask = data.train_mask if split == "train" else data.val_mask

    with torch.no_grad():
        # Predict on the nodes in the split
        logits = model(data)[mask]
        labels = data.credibility[mask]

        probabilities = torch.softmax(logits, dim=-1)
        predicted_classes = torch.argmax(probabilities, dim=-1)

        loss = cross_entropy(logits, labels)

        accuracy = accuracy_score(labels, predicted_classes)
        precision = precision_score(labels, predicted_classes)
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

def train(model: GCN, data, optimizer: optim.Optimizer, epochs: int) -> None:
    for _ in tqdm(range(epochs), desc="Training"):
        train_one_epoch(model, data, optimizer) 
        
        train_metrics = evaluate(model, data, split="train")

        val_metrics = evaluate(model, data, split="val")
        
        wandb.log({"Train": train_metrics, "Val": val_metrics}, step=wandb.run.step)
