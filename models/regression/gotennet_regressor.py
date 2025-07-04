from gotennet import GotenNetWrapper
from gotennet.models.components.layers import CosineCutoff

GotenNetRegressor = GotenNetWrapper(
    n_atom_basis=32,
    n_interactions=2,
    cutoff_fn=CosineCutoff(5.0),
    num_heads=2,
    n_rbf=4
    # Add other parameters as needed, or use defaults
) 