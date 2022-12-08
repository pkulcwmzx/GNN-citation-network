import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SplineConv


class SplineCNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_classes, dropout):
        super(SplineCNN, self).__init__()
        self.dropout = dropout
        self.layer_1 = SplineConv(input_dim, hidden_dim, dim=1, kernel_size=2, aggr="add", cached=True)
        self.layer_2 = SplineConv(hidden_dim, num_classes, dim=1, kernel_size=2, aggr="add", cached=True)

    def reset_parameters(self):
        self.layer_1.reset_parameters()
        self.layer_2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer_1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer_2(x, edge_index)
        return F.log_softmax(x, dim=1)
