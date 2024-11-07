import networkx as nx
import wandb
from torch import optim
from torch.nn import CrossEntropyLoss
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_networkx

from model.gcn import GCN
from utils.processor import train

# Load graph
graph = nx.read_gexf('data/business_users.gexf')

data = from_networkx(graph, group_node_attrs=["betweenness", "clustering", "degree"])

dataloader = DataLoader([data], shuffle=True)

# Create the model
model = GCN(in_channels=data.num_features, hidden_channels=16, out_channels=2)

optimizer = optim.Adam(model.parameters(), lr=0.01)

criterion = CrossEntropyLoss()

wandb.init(project="FIND", name="Initial", tags=("GCN",))

# Train the model
train(
    model=model,
    dataloader=dataloader,
    optimizer=optimizer,
    criterion=criterion,
    epochs=100
)