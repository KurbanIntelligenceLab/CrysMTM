import torch.nn as nn
from models.regression.equiformer.graph_attention_transformer import GraphAttentionTransformer

class EquiformerRegressor(nn.Module):
    def __init__(self, num_targets=1, **kwargs):
        super().__init__()
        self.equiformer = GraphAttentionTransformer(
            irreps_in='5x0e',
            irreps_node_embedding='32x0e+16x1e+8x2e',
            num_layers=2,
            irreps_node_attr='1x0e',
            irreps_sh='1x0e+1x1e+1x2e',
            max_radius=4.0,
            number_of_basis=32,
            fc_neurons=[16, 16],
            irreps_feature='128x0e',
            irreps_head='8x0e+4x1e+2x2e',
            num_heads=1,
            irreps_pre_attn=None,
            rescale_degree=False,
            nonlinear_message=False,
            irreps_mlp_mid='32x0e+16x1e+8x2e',
            norm_layer='layer',
            alpha_drop=0.05,
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
