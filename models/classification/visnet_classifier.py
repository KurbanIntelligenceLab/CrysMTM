import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.models import ViSNet

DEFAULT_MODEL_NAME = "visnet"


class ViSNetEmbedding(nn.Module):
    """Custom ViSNet that returns per-atom embeddings instead of final output."""

    def __init__(self, hidden_channels=16, num_layers=2, cutoff=5.0):
        super(ViSNetEmbedding, self).__init__()
        # Create ViSNet with minimal parameters for faster training
        self.visnet = ViSNet(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            cutoff=cutoff,
            max_num_neighbors=32,
            num_rbf=8,  # Reduced from default 32
            num_heads=4,  # Reduced from default 8
            vertex=False,  # Simpler version without vertex features
        )

    def forward(self, z, pos, batch):
        # Get the embeddings before the final output layers
        batch = torch.zeros_like(z) if batch is None else batch

        # Use the representation model to get per-atom embeddings
        x, vec = self.visnet.representation_model(z, pos, batch)

        # Return the scalar features (per-atom embeddings)
        return x


class ViSNetClassifier(nn.Module):
    def __init__(self, hidden_channels=16, num_layers=2, cutoff=5.0, num_classes=3):
        super(ViSNetClassifier, self).__init__()
        self.visnet = ViSNetEmbedding(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            cutoff=cutoff,
        )
        # Simple classification head
        embedding_size = hidden_channels
        self.classifier = nn.Sequential(nn.Linear(embedding_size, num_classes))

    def forward(self, data):
        # Get molecular representation from ViSNet (per-atom embeddings)
        x = self.visnet(data.z, data.pos, data.batch)
        # Global mean pooling to get per-molecule embeddings
        x = global_mean_pool(x, data.batch)
        # Classification
        return self.classifier(x)
