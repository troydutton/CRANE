import networkx as nx
import optuna
from torch import nn, optim
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.utils.convert import from_networkx
from tqdm import tqdm

from model.gcn import GCN
from utils.misc import set_random_seed
from utils.processor import evaluate, train_one_epoch

# Load graph
graph = nx.read_gexf('data/reddit.gexf')

# Convert to PyTorch Geometric data object
embed_dim = 384
data = from_networkx(graph, group_node_attrs=["betweenness", "clustering", "degree"])

data = RandomNodeSplit(num_val=0.25, num_test=0, key="credibility")(data)

frequencies = (data.credibility > 0.7).to(int).bincount(minlength=2)
weights = frequencies.sum() / (2 * frequencies)

criterion = nn.CrossEntropyLoss(weight=weights)

def objective(trial: optuna.Trial) -> float:
    set_random_seed(42)

    # Dropout
    hidden_channels = trial.suggest_int("hidden_channels", 16, 512)
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    epochs = trial.suggest_int("epochs", 10, 100)

    # Initialize the model
    model = GCN(in_channels=data.num_features, hidden_channels=hidden_channels, out_channels=2)

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    try:
        for epoch in tqdm(range(epochs), desc=f"Trial {trial.number}"):
            train_one_epoch(model, data, optimizer, criterion, verbose=False) 
            
            val_metrics = evaluate(model, data, criterion, split="val", credibility_threshold=0.7)

            trial.report(val_metrics["Metric"]["AUC"], epoch)
    except ValueError: # Model training was too unstable
        return 0
        

    return val_metrics["Metric"]["AUC"]

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")

    study.optimize(objective, n_trials=100, gc_after_trial=True)

    trial = study.best_trial

    print(f"\nTrial #{trial.number} AUC: {trial.value}")
    for key, value in trial.params.items():
        print(f"{key}: {value}")