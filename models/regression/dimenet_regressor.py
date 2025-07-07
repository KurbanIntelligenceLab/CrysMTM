from models.classification.dimenet_classifier import DimeNetPlusPlusNoSparse

DEFAULT_MODEL_NAME = "dimenet++"

DimeNetModel = DimeNetPlusPlusNoSparse(
    hidden_channels=8,
    out_channels=1,  # Regression output
    num_blocks=1,
    int_emb_size=16,
    basis_emb_size=2,
    out_emb_channels=16,
    num_spherical=3,
    num_radial=3,
    cutoff=5.0,
    envelope_exponent=5,
    num_before_skip=1,
    num_after_skip=1,
    num_output_layers=1,
)
