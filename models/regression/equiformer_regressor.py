import torch.nn as nn
from models.regression.equiformer.graph_attention_transformer import GraphAttentionTransformer

class EquiformerRegressor(nn.Module):
    def __init__(self, num_targets=1, **kwargs):
        super().__init__()
        # You can adjust these parameters as needed for your dataset
        self.equiformer = GraphAttentionTransformer(
            irreps_in='5x0e',  # for QM9, adjust for your dataset
            irreps_node_embedding='128x0e+64x1e+32x2e',
            num_layers=6,
            irreps_node_attr='1x0e',
            irreps_sh='1x0e+1x1e+1x2e',
            max_radius=5.0,
            number_of_basis=128,
            fc_neurons=[64, 64],
            irreps_feature='512x0e',
            irreps_head='32x0e+16x1e+8x2e',
            num_heads=4,
            irreps_pre_attn=None,
            rescale_degree=False,
            nonlinear_message=False,
            irreps_mlp_mid='128x0e+64x1e+32x2e',
            norm_layer='layer',
            alpha_drop=0.2,
            proj_drop=0.0,
            out_drop=0.0,
            drop_path_rate=0.0,
            mean=None,
            std=None,
            scale=None,
            atomref=None,
            **kwargs
        )
        self.readout = nn.Linear(1, num_targets)  # Equiformer outputs [batch, 1]

    def forward(self, data):
        # data: PyG Data object with .pos, .batch, .node_atom, etc.
        # You may need to add .node_atom to your Data object (atomic numbers)
        out = self.equiformer(
            f_in=None,  # or torch.ones(data.pos.shape[0], 5) if needed
            pos=data.pos,
            batch=data.batch,
            node_atom=data.node_atom
        )
        return self.readout(out)
