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