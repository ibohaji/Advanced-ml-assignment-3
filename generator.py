import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, to_dense_batch
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool
import networkx as nx
from torch_geometric.utils import to_networkx
import pickle 
from torch_geometric.data import Data




torch.manual_seed(42)
np.random.seed(42)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

dataset = TUDataset(root='./data/', name='MUTAG')
node_feature_dim = dataset.num_node_features

# Print dataset information
print(f"Dataset: {dataset}")
print(f"Number of graphs: {len(dataset)}")
print(f"Number of node features: {node_feature_dim}")
print(f"Number of classes: {dataset.num_classes}")

# Get statistics about the dataset
graph_sizes = []
edge_counts = []
for graph in dataset:
    graph_sizes.append(graph.num_nodes)
    edge_counts.append(graph.num_edges)

max_nodes = max(graph_sizes)
avg_nodes = sum(graph_sizes) / len(graph_sizes)
avg_edges = sum(edge_counts) / len(edge_counts)

print(f"Maximum number of nodes: {max_nodes}")
print(f"Average number of nodes: {avg_nodes:.2f}")
print(f"Average number of edges: {avg_edges:.2f}")

# Split into training and validation
rng = torch.Generator().manual_seed(0)
train_dataset, validation_dataset, test_dataset = random_split(dataset, (100, 44, 44), generator=rng)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)


# Function to convert generated dense adjacency matrices to edge_index format
def dense_to_sparse(adj, node_feat, threshold=0.5):
    batch_size = adj.size(0)
    
    # Initialize lists to store the results for each graph in the batch
    batch_edge_indices = []
    batch_xs = []
    batch_sizes = []
    
    for i in range(batch_size):
        # Get adjacency matrix and node features for this graph
        adj_i = adj[i]
        feat_i = node_feat[i]
        
        # Apply a more aggressive threshold for clearer edges
        binary_adj = (adj_i > threshold).float()
        
        # Determine which nodes are actually used (have at least one connection)
        node_mask = (binary_adj.sum(dim=1) > 0) | (feat_i.sum(dim=1) > 0.1)
        num_nodes = node_mask.sum().item()
        
        if num_nodes < 2:  # Ensure at least 2 nodes
            num_nodes = 2
            node_mask[:num_nodes] = True
        
        # Extract the used parts of the adjacency matrix and features
        adj_i = binary_adj[node_mask][:, node_mask]
        feat_i = feat_i[node_mask]
        
        # Convert to edge_index format
        edge_index = adj_i.nonzero().t()
        
        # Store results
        batch_edge_indices.append(edge_index)
        batch_xs.append(feat_i)
        batch_sizes.append(num_nodes)
    
    return batch_edge_indices, batch_xs, batch_sizes



# --------------------- MODEL ---------------------


class GraphVAE(nn.Module):
    def __init__(self, max_nodes, node_feat_dim, latent_dim=64):
        super().__init__()
        self.max_nodes = max_nodes
        self.node_feat_dim = node_feat_dim
        self.latent_dim = latent_dim
        # Encoder
        self.enc_conv1 = GCNConv(node_feat_dim, 32)
        self.enc_conv2 = GCNConv(32, 64)
        self.enc_conv3 = GCNConv(64, 128)
        self.mu = nn.Linear(128, latent_dim)
        self.logvar = nn.Linear(128, latent_dim)
        
         # Decoder
        self.dec_lin = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.max_nodes * 32)  # Output is node embedding matrix
        )


    def encode(self, x, edge_index, batch):
        x = F.relu(self.enc_conv1(x, edge_index))
        x = F.relu(self.enc_conv2(x, edge_index))
        x = F.relu(self.enc_conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.mu(x), self.logvar(x)

    
    def decode(self, z):
        node_repr = self.dec_lin(z).view(-1, self.max_nodes, 32)
        adj_logits = torch.matmul(node_repr, node_repr.transpose(1, 2))
        adj_logits = adj_logits - torch.diag_embed(torch.diagonal(adj_logits, dim1=1, dim2=2))  # No self-loops
        return torch.sigmoid(adj_logits)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x, edge_index, batch):
        mu, logvar = self.encode(x, edge_index, batch)
        z = self.reparameterize(mu, logvar)
        adj_recon = self.decode(z)
        return adj_recon, mu, logvar

    def sample(self, num_samples, device):
        z = torch.randn(num_samples, self.latent_dim, device=device)
        adj_logits  = self.decode(z)
        adj_sampled = torch.bernoulli(adj_logits)
        return adj_sampled
    

# --------------------- TRAINING FUNCTIONS ---------------------

def vae_loss(adj_recon, adj_truth, mu, logvar):
    # Reconstruction loss (cross entropy for edges)
    BCE = F.binary_cross_entropy(adj_recon, adj_truth)
    
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD*0.001  # Weight KL term




# --------------------- MAIN FUNCTION ---------------------
def main():
    # Hyperparameters
    latent_dim = 128 * 2
    num_epochs = 400
    max_nodes = 17
    node_feat_dim = 7

    processed_dataset = []
    for data in dataset:
        dense_adj = to_dense_adj(data.edge_index, max_num_nodes=max_nodes)[0]
        data.adj = dense_adj
        processed_dataset.append(data)
    
    train_dataset, val_dataset, test_dataset = random_split(processed_dataset, (100, 44, 44), generator=torch.Generator().manual_seed(42))

    batch_size= 5
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model
    model = GraphVAE(max_nodes=max_nodes, node_feat_dim=node_feat_dim, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    # Training tracking
    train_losses = []
    val_losses = []
    best_loss = float('inf')

    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_train_loss = 0
        for batch in train_loader:
            x, edge_index, batch_idx = batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device)
            adj_truth = torch.stack([g.adj for g in batch.to_data_list()]).to(device)
            
            optimizer.zero_grad()
            adj_recon, mu, logvar = model(x, edge_index, batch_idx)
            loss = vae_loss(adj_recon, adj_truth, mu, logvar)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item() * len(batch) / len(train_dataset)
        
        # Validation
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x, edge_index, batch_idx = batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device)
                adj_truth = torch.stack([g.adj for g in batch.to_data_list()]).to(device)
                
                adj_recon, mu, logvar = model(x, edge_index, batch_idx)
                loss = vae_loss(adj_recon, adj_truth, mu, logvar)
                epoch_val_loss += loss.item() * len(batch) / len(val_dataset)
        
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}')
            print(f'Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}')
        
        # Save best model
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            torch.save(model.state_dict(), 'best_vae_model.pt')




    # Generate 1000 samples
    model.load_state_dict(torch.load('best_vae_model.pt'))
    model.eval()
    
    generated_graphs = []
    with torch.no_grad():
        for _ in range(1000):
            # Sample new graph
            adj = model.sample(1, device).cpu().squeeze()
            
            # Convert to edge_index format
            edge_index = (adj > 0.5).nonzero().t()
            
            # Create NetworkX graph
            if edge_index.size(1) > 0:
                data = Data(edge_index=edge_index)
                G = to_networkx(data)
                generated_graphs.append(G)
    
    with open('generated_graphs.pkl', 'wb') as f:
        pickle.dump(generated_graphs, f)
    
    print(f"Generated {len(generated_graphs)} valid graphs")

if __name__ == "__main__":
    main()