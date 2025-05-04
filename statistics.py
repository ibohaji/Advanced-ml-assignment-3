# evaluate_novelty_uniqueness.py
import pickle
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from torch_geometric.loader import DataLoader
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np 


with open("generated_graphs.pkl", "rb") as f:
    gen_graphs = pickle.load(f)  # List of 1000 NetworkX graphs

with open("baseline_samples.pkl", "rb") as f:
    baseline_graphs = pickle.load(f)  # List of 1000 NetworkX graphs

dataset = TUDataset(root='./data', name='MUTAG')
train_dataset = dataset[:100]

# Convert training graphs to NetworkX
train_graphs = [to_networkx(data, to_undirected=True) for data in train_dataset]

# Compute WL hashes
def wl_hashes(graphs):
    return [weisfeiler_lehman_graph_hash(g) for g in graphs]


train_hashes = set(wl_hashes(train_graphs))
sample_hashes = wl_hashes(gen_graphs)

unique_sample_hashes = set(sample_hashes)

# Metrics
n_total = len(sample_hashes)
n_unique = len(unique_sample_hashes)
n_novel = sum(h not in train_hashes for h in sample_hashes)
n_novel_unique = len(unique_sample_hashes - train_hashes)

# Report
print(f"{'Metric':<20} {'Percentage':>10}")
print(f"{'-'*30}")
print(f"{'Novel':<20} {n_novel / n_total * 100:>9.1f}%")
print(f"{'Unique':<20} {n_unique / n_total * 100:>9.1f}%")
print(f"{'Novel + Unique':<20} {n_novel_unique / n_total * 100:>9.1f}%")



def plot_graph_statistics(train_graphs, generated_graphs, baseline_graphs=None):
    def collect_stats(graphs):
        degrees = []
        clustering = []
        centrality = []
        for G in graphs:
            degrees += [d for _, d in G.degree()]
            clustering += list(nx.clustering(G).values())
            try:
                c = nx.eigenvector_centrality_numpy(G)
                centrality += list(c.values())
            except nx.NetworkXException:
                continue  # skip graphs that fail
        return degrees, clustering, centrality

    # Gather statistics
    train_stats = collect_stats(train_graphs)
    gen_stats = collect_stats(generated_graphs)
    base_stats = collect_stats(baseline_graphs) if baseline_graphs else ([], [], [])

    # Define titles and figure
    metric_names = ["Node Degree", "Clustering Coefficient", "Eigenvector Centrality"]
    fig, axs = plt.subplots(3, 3, figsize=(15, 12))

    for i in range(3):
        # Shared bin edges across all 3 distributions for this metric
        all_data = train_stats[i] + gen_stats[i] + base_stats[i]
        bins = np.histogram_bin_edges(all_data, bins='auto')

        axs[i, 0].hist(train_stats[i], bins=bins, color='blue', alpha=0.7)
        axs[i, 0].set_title(f"Train {metric_names[i]}")

        axs[i, 1].hist(gen_stats[i], bins=bins, color='green', alpha=0.7)
        axs[i, 1].set_title(f"Generated {metric_names[i]}")

        axs[i, 2].hist(base_stats[i], bins=bins, color='orange', alpha=0.7)
        axs[i, 2].set_title(f"Baseline {metric_names[i]}")

        for j in range(3):
            axs[i, j].set_xlabel(metric_names[i])
            axs[i, j].set_ylabel("Count")

    plt.tight_layout()
    plt.show()
    fig.savefig("plot.png")


# Plot graph statistics 


plot_graph_statistics(train_graphs, gen_graphs, baseline_graphs) 