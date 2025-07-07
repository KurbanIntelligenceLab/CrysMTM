from faenet import FAENet

FAENetRegressor = FAENet(
    cutoff=5.0,
    act="silu",
    preprocess="base_preprocess",
    complex_mp=False,
    max_num_neighbors=20,
    num_gaussians=8,
    num_filters=32,
    hidden_channels=32,
    tag_hidden_channels=8,
    pg_hidden_channels=8,
    phys_hidden_channels=0,
    phys_embeds=False,  # Disable physics-aware embeddings
    num_interactions=2,
    mp_type="base",
    graph_norm=True,
    second_layer_MLP=False,
    skip_co="add",
    energy_head=None,
    regress_forces=None,
    force_decoder_type="mlp",
    force_decoder_model_config={"hidden_channels": 32},
)
