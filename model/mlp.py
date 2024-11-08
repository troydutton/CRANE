from typing import List

from torch import Tensor, nn
from torch_geometric.data import Data


class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout: float = 0.0) -> None:
        super().__init__()

        layers = list()

        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))

            input_dim = dim

        layers.append(nn.Linear(input_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, graph: Data) -> Tensor:
        return self.mlp(graph.x)
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)