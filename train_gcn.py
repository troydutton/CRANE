import networkx as nx
from torch import nn, optim
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.utils.convert import from_networkx

import wandb
from model.gcn import GCN
from utils.misc import set_random_seed
from utils.processor import train

set_random_seed(42)

# Load graph
print("Loading graph...")
graph = nx.read_gexf('data/weighted_temp.gexf')

# Unweighted
data = from_networkx(graph, group_node_attrs=["betweenness", "clustering", "degree"])

# Unweighted w/ embeddings
# data = from_networkx(graph, group_node_attrs=["betweenness", "clustering", "degree", *[f"embedding_{i}" for i in range(embed_dim := 384)]])

# Weighted
# data = from_networkx(graph, group_node_attrs=["betweenness", "clustering", "degree"], group_edge_attrs=["weight"])

data = RandomNodeSplit(num_val=0.25, num_test=0, key="credibility")(data)

frequencies = (data.credibility > 0.5).to(int).bincount(minlength=2)
weights = frequencies.sum() / (2 * frequencies)

# Initialize the model
model = GCN(in_channels=data.num_features, hidden_channels=16, out_channels=2)

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-2)

criterion = nn.CrossEntropyLoss(weight=weights)

# Initialize WandB
wandb.init(project="CRANE", name="GCN", tags=("GCN",))

# Train the model
train(
    model=model,
    data=data,
    optimizer=optimizer,
    criterion=criterion,
    epochs=100,
    credibility_threshold=0.5
)