from models.classification.dimenet_classifier import DimeNetPlusPlusNoSparse

DEFAULT_MODEL_NAME = "dimenet++"

DimeNetModel = DimeNetPlusPlusNoSparse(
    hidden_channels=16,
    out_channels=1,  # Regression output
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
) 