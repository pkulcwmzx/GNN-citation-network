import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv

class GAT(nn.Module):

    def __init__(self, input_dim, hidden_dim, heads, dropout, num_classes):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.layer_1 = GATConv(input_dim, hidden_dim, heads, dropout=dropout)
        self.layer_2 = GATConv(hidden_dim * heads, num_classes, 8, dropout=dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer_1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer_2(x, edge_index)
        return F.log_softmax(x, dim=1)
