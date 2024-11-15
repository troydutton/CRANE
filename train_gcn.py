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
graph = nx.read_gexf('data/reddit.gexf')

# Convert to PyTorch Geometric data object
embed_dim = 384
data = from_networkx(graph, group_node_attrs=["betweenness", "clustering", "degree", *[f"embedding_{i}" for i in range(embed_dim)]])
# data = from_networkx(graph, group_node_attrs=["betweenness", "clustering", "degree"])

data = RandomNodeSplit(num_val=0.25, num_test=0, key="credibility")(data)

frequencies = (data.credibility > 0.5).to(int).bincount(minlength=2)
weights = frequencies.sum() / (2 * frequencies)

# Initialize the model
model = GCN(in_channels=data.num_features, hidden_channels=16, out_channels=2)

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-2)

criterion = nn.CrossEntropyLoss(weight=weights)

# Initialize WandB
wandb.init(project="FIND", name="GCN Post Embeddings", tags=("GCN",))

# Train the model
train(
    model=model,
    data=data,
    optimizer=optimizer,
    criterion=criterion,
    epochs=50,
    credibility_threshold=0.5
)