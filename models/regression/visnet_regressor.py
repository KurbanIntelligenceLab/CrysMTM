import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.models import ViSNet

DEFAULT_MODEL_NAME = "visnet"

class ViSNetEmbedding(nn.Module):
    """Custom ViSNet that returns per-atom embeddings instead of final output."""
    def __init__(self, hidden_channels=16, num_layers=2, cutoff=5.0):
        super(ViSNetEmbedding, self).__init__()
        self.visnet = ViSNet(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            cutoff=cutoff,
            max_num_neighbors=32,
            num_rbf=8,
            num_heads=4,
            vertex=False,
        )
    def forward(self, z, pos, batch):
        batch = torch.zeros_like(z) if batch is None else batch
        x, vec = self.visnet.representation_model(z, pos, batch)
        return x

class ViSNetRegressor(nn.Module):
    def __init__(self, hidden_channels=16, num_layers=2, cutoff=5.0):
        super(ViSNetRegressor, self).__init__()
        self.visnet = ViSNetEmbedding(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            cutoff=cutoff,
        )
        embedding_size = hidden_channels
        self.regressor = nn.Linear(embedding_size, 1)
    def forward(self, data):
        x = self.visnet(data.z, data.pos, data.batch)
        x = global_mean_pool(x, data.batch)
        return self.regressor(x) 