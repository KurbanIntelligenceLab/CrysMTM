from egnn_pytorch.egnn_pytorch_geometric import EGNN_Sparse_Network

EGNNRegressor = EGNN_Sparse_Network(
    n_layers=3,
    feats_dim=1, 
    pos_dim=3,
    m_dim=16,
    update_coors=True,
    update_feats=True,
    norm_feats=True,
    norm_coors=False,
    dropout=0.0,
    coor_weights_clamp_value=2.0,
) 