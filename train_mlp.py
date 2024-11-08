import networkx as nx
from torch import optim
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.utils.convert import from_networkx

import wandb
from model.mlp import MultiLayerPerceptron
from utils.misc import set_random_seed
from utils.processor import train

set_random_seed(42)

# Load graph
graph = nx.read_gexf('data/business_users.gexf')

# Convert to PyTorch Geometric data object
data = from_networkx(graph, group_node_attrs=["betweenness", "clustering", "degree"])

data = RandomNodeSplit(num_val=0.25, num_test=0, key="credibility")(data)

# Initialize the model
model = MultiLayerPerceptron(input_dim=data.num_features, hidden_dims=[16, 16], output_dim=2)

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Initialize WandB
wandb.init(project="FIND", name="MLP", tags=("MLP",))

# Train the model
train(
    model=model,
    data=data,
    optimizer=optimizer,
    epochs=100
)