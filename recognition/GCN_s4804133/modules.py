# use torch instead of numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """Baseline model used for comparison; ignores connections between pages and only classifies based on the individual page features.
    Layer order (graph layer, ReLU, dropout, graph layer) is adapted for an MLP from the two-layer GCN in Kipf and Welling (2017) and their reference code at https://github.com/tkipf/pygcn. Specific values, configurations and transformations used are my own. """

    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim) # number of input features to hidden layer (128 for this dataset)
        self.fc2 = nn.Linear(hidden_dim, out_dim) # number of hidden layer features to output layer (4 classes)
        self.dropout = dropout # half the values are zeroed out on each training pass

    def forward(self, x, edge_index=None):
        x = F.relu(self.fc1(x)) # apply Relu activation to hidden layer output (sets negative values to 0)
        x = F.dropout(x, p=self.dropout, training=self.training) # zero out half of the 128 values during training
        return self.fc2(x)  
    
class GCNLayer(nn.Module):
    """One graph convolution: out = D^-1/2 (A + I) D^-1/2 X W + b, using edge lists. Weight initialisation (Xavier uniform) and the X @ W layer structure are adapted from the example layer on the "Some code - Matrix" slide of the COMP3710 GNN lecture (Kulatilleke and Chandra, 14 Oct 2026). The edge-list implementation and the actual inputs, and content of each variable is my own."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim)) # bias vector all zeros to
        nn.init.xavier_uniform_(self.weight) # initialise weights with Xavier uniform distribution; inspired by 

    def forward(self, x, edge_index):
        num_nodes = x.size(0)

        # Apply layer's learned weights to the input features using matrix mult.
        h = x @ self.weight

        # Add self-loops so each node includes its own features in the aggregation
        loops = torch.arange(num_nodes, device=x.device)
        src = torch.cat([edge_index[0], loops])
        dst = torch.cat([edge_index[1], loops])

        # Degree matrix computes a weight for each edge + symmetric normalisation
        ones = torch.ones(dst.size(0), device=x.device)
        deg = torch.zeros(num_nodes, device=x.device).index_add_(0, dst, ones)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]

        # Message passing, each edge sends its source's weighted features to its destination, summed per node
        out = torch.zeros(num_nodes, h.size(1), device=x.device)
        out.index_add_(0, dst, h[src] * norm.unsqueeze(1))
        return out + self.bias # add leanred bias vectr to every row