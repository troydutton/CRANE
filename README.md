# CRANE: Credibility Rating and Network Evaluation

CRANE leverages user interaction networks and content embeddings to evaluate credibility, addressing the widespread issue of misinformation on social media platforms.

## Installation

1. Create a conda environment and install the required dependencies:
```bash
mamba env create -f environment.yaml
```

2. Activate the environment:
```bash
conda activate crane 
```

3. Log in to W&B:
```
wandb login
```

## Usage

1. Create the user-to-user network:
```
python create_network.py
```

2. Optionally, if you are training a GCN, select a graph type and corresponding graph path:

```
GRAPH_PATH = "data/unweighted.gexf" OR "data/weighted.gexf"
```
```
GRAPH_TYPE = "unweighted" OR "embeddings" OR "weighted"
```
3. Train a model to predict user credibility:
```
python train_gcn.py
```

OR

```
python train_mlp.py
```