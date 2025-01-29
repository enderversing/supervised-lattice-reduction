import numpy as np
from load import LatticeDataset
from torch_geometric.data import Dataset, Data
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
import torch

input = np.load("unreduced_30m_random.npy")
output = np.load("reduced_30m_random.npy")


class GCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = nn.functional.relu(x)
        x = self.conv2(x, edge_index)
        return x


dataset = LatticeDataset(input, output)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = GCNModel(input_dim=dataset[0].x.size(-1), hidden_dim=64, output_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    for batch in dataloader:
        optimizer.zero_grad()
        output = model(batch)
        loss = nn.functional.mse_loss(output, batch.y)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")
