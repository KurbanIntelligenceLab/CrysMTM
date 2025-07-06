import os
import torch
import torch.nn as nn
import pandas as pd
import pickle
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import global_mean_pool, knn_graph

from dataloaders.regression_dataloader import RegressionLoader
from configs.regression_config import (
    SEEDS, TARGET_PROPERTIES, BASE_DIR, MAX_ROTATIONS, NORMALIZE_LABELS, NORMALIZATION_METHOD, ID_TEMPS, BATCH_SIZE, OOD_TEMPS
)
TEST_TEMPS = ID_TEMPS + OOD_TEMPS
def load_egnn_model(target_name, seed, device):
    """Load trained EGNN model."""
    model_path = os.path.join("results", f"egnn/{target_name}", str(seed), "best_model.pth")
    
    # Load normalizer if available
    normalizer_path = os.path.join("results", f"egnn/{target_name}", str(seed), "normalizer.pkl")
    normalizer = None
    if os.path.exists(normalizer_path):
        with open(normalizer_path, 'rb') as f:
            normalizer = pickle.load(f)
    
    from models.regression.egnn_regressor import EGNNRegressor
    model = EGNNRegressor.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Load regression head if available
    reg_head_path = os.path.join("results", f"egnn/{target_name}", str(seed), "best_reg_head.pth")
    if os.path.exists(reg_head_path):
        reg_head = nn.Linear(1, 1).to(device)
        reg_head.load_state_dict(torch.load(reg_head_path, map_location=device))
    else:
        reg_head = None
    
    return model, reg_head, normalizer

def evaluate_egnn_detailed(model_info, dataloader, device, target_name, seed):
    """Evaluate EGNN model and return detailed predictions with metadata."""
    model, reg_head, normalizer = model_info
    model.eval()
    if reg_head:
        reg_head.eval()
    
    predictions = []
    targets = []
    metadata = []
    
    # Track the actual dataset indices
    dataset_idx_counter = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device)
            
            # EGNN specific logic - use EXACT same forward pass logic as train_egnn.py
            feats = batch.z.unsqueeze(-1).float()  # [num_nodes, 1]
            x = torch.cat([batch.pos, feats], dim=1)  # [num_nodes, 4]
            edge_index = batch.edge_index
            if edge_index is None:
                edge_index = knn_graph(batch.pos, k=8, batch=batch.batch)
            num_edges = edge_index.size(1)
            dummy_edge_attr = x.new_zeros((num_edges, 0))
            out = model(x, edge_index, batch.batch, edge_attr=dummy_edge_attr)
            node_feats = out[:, 3:]  # [num_nodes, feats_dim]
            pooled = global_mean_pool(node_feats, batch.batch)
            outputs = reg_head(pooled)
            
            # Get target for the specific property
            target_index = TARGET_PROPERTIES.index(target_name)
            labels = batch.y
            if labels.dim() == 1:
                target = labels.unsqueeze(1)
            else:
                target = labels[:, target_index].unsqueeze(1)
            

            
            # Convert to original scale if normalizer is used
            if normalizer is not None:
                # For evaluation, we want to compare predictions and targets in the same scale
                # Since the model was trained on normalized data, let's compare in normalized scale
                pred_batch = outputs.cpu().numpy()  # Keep normalized
                target_batch = target.cpu().numpy()  # Keep normalized
            else:
                pred_batch = outputs.cpu().numpy()
                target_batch = target.cpu().numpy()
            
            # Extract metadata - one entry per sample in the batch
            # Use batch['temperature'] if available for batch size
            if hasattr(batch, 'temperature') or hasattr(batch, 'temperatures'):
                # If batch is a dict-like object (unlikely for PyG, but just in case)
                batch_size = len(getattr(batch, 'temperature', getattr(batch, 'temperatures', [])))
            elif hasattr(batch, 'batch') and batch.batch is not None:
                batch_size = batch.batch.max().item() + 1
            else:
                batch_size = len(target_batch.flatten())
            # Only use the first batch_size predictions/targets
            predictions.extend(pred_batch.flatten()[:batch_size])
            targets.extend(target_batch.flatten()[:batch_size])

            for i in range(batch_size):
                # Try to get metadata from dataset
                dataset_idx = dataset_idx_counter + i
                if hasattr(dataloader.dataset, 'data') and dataset_idx < len(dataloader.dataset.data):
                    entry = dataloader.dataset.data[dataset_idx]
                    temperature = entry.get('temperature', None)
                    composition = entry.get('phase', None)
                else:
                    temperature = None
                    composition = None
                if temperature is None or composition is None:
                    print(f"WARNING: Missing metadata at batch {batch_idx}, sample {i}")
                metadata.append({
                    'model': 'egnn',
                    'target': target_name,
                    'seed': seed,
                    'batch_idx': i,
                    'sample_idx': dataset_idx,
                    'temperature': temperature,
                    'composition': composition
                })
            # Update the counter for the next batch
            dataset_idx_counter += batch_size
    
    return predictions, targets, metadata

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    output_dir = "results/detailed_analysis"
    os.makedirs(output_dir, exist_ok=True)
    combined_path = os.path.join(output_dir, "egnn_predictions.csv")
    first_write = not os.path.exists(combined_path)
    
    for target_name in TARGET_PROPERTIES:
        for seed in SEEDS:
            try:
                model_path = os.path.join("results", f"egnn/{target_name}", str(seed), "best_model.pth")
                if not os.path.exists(model_path):
                    print(f"Model not found: {model_path}")
                    continue
                

                
                model_info = load_egnn_model(target_name, seed, device)
                model, reg_head, normalizer = model_info
                
                # Use appropriate dataloader logic for EGNN
                id_dataset = RegressionLoader(
                    label_dir=BASE_DIR,
                    temperature_filter=lambda temp: temp in TEST_TEMPS,
                    modalities=["xyz", "element"],
                    max_rotations=MAX_ROTATIONS,
                    as_pyg_data=True,
                    normalize_labels=NORMALIZE_LABELS,
                    normalization_method=NORMALIZATION_METHOD,
                    fit_normalizer_on_data=False,
                )
                
                # Set the fitted normalizer for ID dataset
                if normalizer is not None:
                    id_dataset.set_normalizer(normalizer)
                

                
                dataloader = PyGDataLoader(id_dataset, batch_size=BATCH_SIZE, shuffle=False)
                
                predictions, targets, metadata = evaluate_egnn_detailed(
                    model_info, dataloader, device, target_name, seed
                )
                
                df = pd.DataFrame({
                    'model': 'egnn',
                    'property': target_name,
                    'seed': seed,
                    'prediction': predictions,
                    'actual': targets,
                    'temperature': [m.get('temperature') for m in metadata],
                    'composition': [m.get('composition') for m in metadata]
                })
                
                df.to_csv(combined_path, mode='a', header=first_write, index=False)
                first_write = False
                print(f"Saved results for egnn/{target_name}/{seed} to {combined_path}")
                
            except Exception as e:
                print(f"Error evaluating egnn/{target_name}/{seed}: {e}")
                continue
    
    print(f"All EGNN results saved to {combined_path}")

if __name__ == "__main__":
    main() 