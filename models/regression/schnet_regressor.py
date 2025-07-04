from torch_geometric.nn import SchNet

DEFAULT_MODEL_NAME = "schnet"

SchnetModel = SchNet(
    hidden_channels=16,
    num_filters=16,
    num_interactions=2,
    num_gaussians=8,
    cutoff=5.0,
    readout="add",
)
