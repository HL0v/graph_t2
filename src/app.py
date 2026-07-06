import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

# 1. Load the Cora Dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')

# 2. Define the Neural Network Architecture
class SimpleGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # First layer maps the 1,433 word features to 16 hidden dimensions
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        # Second layer maps the 16 hidden dimensions to the 7 output classes
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Layer 1: Message Passing + Non-linearity
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Layer 2: Final prediction
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Usage execution
model = SimpleGCN()
data = dataset[0]
output = model(data)