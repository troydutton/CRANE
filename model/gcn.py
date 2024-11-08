import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        super(GCN, self).__init__()
        
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, graph: Data) -> Tensor:
        x, edges = graph.x, graph.edge_index

        # First layer
        x = self.conv1(x, edges)
        x = F.relu(x)

        # Second layer
        x = self.conv2(x, edges)

        return x
