import torch
import torch.nn as nn
from torch_geometric.nn import SchNet, global_mean_pool

DEFAULT_MODEL_NAME = "schnet"


class SchNetEmbedding(nn.Module):
    """Custom SchNet that returns per-molecule embeddings instead of final output."""

    def __init__(
        self,
        hidden_channels=16,
        num_filters=16,
        num_interactions=2,
        num_gaussians=8,
        cutoff=5.0,
    ):
        super(SchNetEmbedding, self).__init__()
        self.schnet = SchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
        )

    def forward(self, z, pos, batch):
        # Get the embeddings before the final linear layers
        batch = torch.zeros_like(z) if batch is None else batch

        h = self.schnet.embedding(z)
        edge_index, edge_weight = self.schnet.interaction_graph(pos, batch)
        edge_attr = self.schnet.distance_expansion(edge_weight)

        for interaction in self.schnet.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        # Apply the first linear layer and activation
        h = self.schnet.lin1(h)
        h = self.schnet.act(h)

        # Return the embeddings before the final linear layer
        return h


class SchNetClassifier(nn.Module):
    def __init__(
        self,
        hidden_channels=16,
        num_filters=16,
        num_interactions=2,
        num_gaussians=8,
        cutoff=5.0,
        num_classes=3,
    ):
        super(SchNetClassifier, self).__init__()
        self.schnet = SchNetEmbedding(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
        )
        # Simple classification head - input size is hidden_channels//2 due to lin1 layer
        embedding_size = hidden_channels // 2
        self.classifier = nn.Sequential(nn.Linear(embedding_size, num_classes))

    def forward(self, data):
        # Get molecular representation from SchNet (per-atom embeddings)
        x = self.schnet(data.z, data.pos, data.batch)
        # Global mean pooling to get per-molecule embeddings
        x = global_mean_pool(x, data.batch)
        # Classification
        return self.classifier(x)
