import torch
import torch.nn as nn
from torch_geometric.nn import DimeNetPlusPlus, radius_graph
from torch_geometric.utils import scatter

DEFAULT_MODEL_NAME = "dimenet"


def triplets(edge_index, num_nodes):
    """Extract triplets from edge_index for DimeNet without using SparseTensor."""
    row, col = edge_index  # j->i

    # Create adjacency list
    adj = [[] for _ in range(num_nodes)]
    for i, (j, k) in enumerate(zip(row, col)):
        adj[j].append((k, i))  # (neighbor, edge_idx)

    # Extract triplets
    idx_i, idx_j, idx_k = [], [], []
    idx_kj, idx_ji = [], []

    for j in range(num_nodes):
        for k, edge_kj in adj[j]:
            for i, edge_ji in adj[j]:
                if i != k:  # Remove i == k triplets
                    idx_i.append(i)
                    idx_j.append(j)
                    idx_k.append(k)
                    idx_kj.append(edge_kj)
                    idx_ji.append(edge_ji)

    if len(idx_i) == 0:
        # Handle empty case
        return (
            col,
            row,
            torch.tensor([], dtype=torch.long, device=edge_index.device),
            torch.tensor([], dtype=torch.long, device=edge_index.device),
            torch.tensor([], dtype=torch.long, device=edge_index.device),
            torch.tensor([], dtype=torch.long, device=edge_index.device),
            torch.tensor([], dtype=torch.long, device=edge_index.device),
        )

    return (
        col,
        row,
        torch.tensor(idx_i, dtype=torch.long, device=edge_index.device),
        torch.tensor(idx_j, dtype=torch.long, device=edge_index.device),
        torch.tensor(idx_k, dtype=torch.long, device=edge_index.device),
        torch.tensor(idx_kj, dtype=torch.long, device=edge_index.device),
        torch.tensor(idx_ji, dtype=torch.long, device=edge_index.device),
    )


class DimeNetPlusPlusNoSparse(DimeNetPlusPlus):
    """DimeNet++ using manual triplets calculation to avoid torch-sparse."""

    def forward(self, z, pos, batch=None):
        edge_index = radius_graph(
            pos, r=self.cutoff, batch=batch, max_num_neighbors=self.max_num_neighbors
        )
        # Use manual triplets function from this file
        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
            edge_index, num_nodes=z.size(0)
        )

        # Calculate distances
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        # Calculate angles
        pos_jk, pos_ij = pos[idx_j] - pos[idx_k], pos[idx_i] - pos[idx_j]
        a = (pos_ij * pos_jk).sum(dim=-1)
        b = torch.cross(pos_ij, pos_jk, dim=1).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        # Embedding block
        x = self.emb(z, rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))

        # Interaction blocks
        for interaction_block, output_block in zip(
            self.interaction_blocks, self.output_blocks[1:]
        ):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            P = P + output_block(x, rbf, i, num_nodes=pos.size(0))

        if batch is None:
            return P.sum(dim=0)
        else:
            return scatter(P, batch, dim=0, reduce="sum")


class DimeNetPlusPlusEmbedding(nn.Module):
    """Custom DimeNet++ that returns per-molecule embeddings instead of final output."""

    def __init__(
        self,
        hidden_channels=16,
        out_channels=16,
        num_blocks=2,
        int_emb_size=32,
        basis_emb_size=4,
        out_emb_channels=32,
        num_spherical=5,
        num_radial=4,
        cutoff=5.0,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=1,
        num_output_layers=2,
    ):
        super(DimeNetPlusPlusEmbedding, self).__init__()
        self.dimenet = DimeNetPlusPlusNoSparse(
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
            int_emb_size=int_emb_size,
            basis_emb_size=basis_emb_size,
            out_emb_channels=out_emb_channels,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
            envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            num_output_layers=num_output_layers,
        )

    def forward(self, z, pos, batch):
        # Use the built-in DimeNetPlusPlus forward, which handles batching correctly
        return self.dimenet(z, pos, batch)


class DimeNetPlusPlusClassifier(nn.Module):
    def __init__(
        self,
        hidden_channels=16,
        out_channels=16,
        num_blocks=2,
        int_emb_size=32,
        basis_emb_size=4,
        out_emb_channels=32,
        num_spherical=5,
        num_radial=4,
        cutoff=5.0,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=1,
        num_output_layers=2,
        num_classes=3,
    ):
        super(DimeNetPlusPlusClassifier, self).__init__()
        self.dimenet = DimeNetPlusPlusEmbedding(
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
            int_emb_size=int_emb_size,
            basis_emb_size=basis_emb_size,
            out_emb_channels=out_emb_channels,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
            envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            num_output_layers=num_output_layers,
        )
        # Simple classification head
        embedding_size = hidden_channels
        self.classifier = nn.Sequential(nn.Linear(embedding_size, num_classes))

    def forward(self, data):
        x = self.dimenet(data.z, data.pos, data.batch)
        return self.classifier(x)
