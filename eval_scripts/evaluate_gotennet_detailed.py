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
def load_gotennet_model(target_name, seed, device):
    """Load trained GotenNet model and regression head."""
    model_path = os.path.join("results", f"regression/gotennet/{target_name}", str(seed), "best_model.pth")
    reg_head_path = os.path.join("results", f"regression/gotennet/{target_name}", str(seed), "best_reg_head.pth")
    normalizer_path = os.path.join("results", f"regression/gotennet/{target_name}", str(seed), "normalizer.pkl")
    normalizer = None
    if os.path.exists(normalizer_path):
        with open(normalizer_path, 'rb') as f:
            normalizer = pickle.load(f)
    from models.regression.gotennet_regressor import GotenNetRegressor
    model = GotenNetRegressor.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    hidden_dim = 32  # n_atom_basis
    reg_head = nn.Linear(hidden_dim, 1).to(device)
    reg_head.load_state_dict(torch.load(reg_head_path, map_location=device))
    return model, reg_head, normalizer

def evaluate_gotennet_detailed(model_info, dataloader, device, target_name, seed):
    """Evaluate GotenNet model and return detailed predictions with metadata."""
    model, reg_head, normalizer = model_info
    model.eval()
    reg_head.eval()
    predictions = []
    targets = []
    metadata = []
    dataset_idx_counter = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device)
            # GotenNet specific logic - uses batch object and returns h, X
            h, X = model(batch)  # h: [num_nodes, hidden_dim]
            pooled = global_mean_pool(h, batch.batch)
            outputs = reg_head(pooled)
            target_index = TARGET_PROPERTIES.index(target_name)
            labels = batch.y
            if labels.dim() == 1:
                target = labels.unsqueeze(1)
            else:
                target = labels[:, target_index].unsqueeze(1)
            if normalizer is not None:
                outputs_orig = torch.tensor(
                    normalizer.inverse_transform(outputs.cpu().numpy()),
                    dtype=torch.float,
                    device=device
                )
                target_orig = torch.tensor(
                    normalizer.inverse_transform(target.cpu().numpy()),
                    dtype=torch.float,
                    device=device
                )
                pred_batch = outputs_orig.cpu().numpy()
                target_batch = target_orig.cpu().numpy()
            else:
                pred_batch = outputs.cpu().numpy()
                target_batch = target.cpu().numpy()
            # Robust batch/metadata alignment
            if hasattr(batch, 'temperature') or hasattr(batch, 'temperatures'):
                batch_size = len(getattr(batch, 'temperature', getattr(batch, 'temperatures', [])))
            elif hasattr(batch, 'batch') and batch.batch is not None:
                batch_size = batch.batch.max().item() + 1
            else:
                batch_size = len(target_batch.flatten())
            predictions.extend(pred_batch.flatten()[:batch_size])
            targets.extend(target_batch.flatten()[:batch_size])
            for i in range(batch_size):
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
                    'model': 'gotennet',
                    'target': target_name,
                    'seed': seed,
                    'batch_idx': i,
                    'sample_idx': dataset_idx,
                    'temperature': temperature,
                    'composition': composition
                })
            dataset_idx_counter += batch_size
    return predictions, targets, metadata

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    output_dir = "results/detailed_analysis"
    os.makedirs(output_dir, exist_ok=True)
    combined_path = os.path.join(output_dir, "gotennet_predictions.csv")
    first_write = not os.path.exists(combined_path)
    for target_name in TARGET_PROPERTIES:
        for seed in SEEDS:
            try:
                model_path = os.path.join("results", f"regression/gotennet/{target_name}", str(seed), "best_model.pth")
                if not os.path.exists(model_path):
                    print(f"Model not found: {model_path}")
                    continue
                model_info = load_gotennet_model(target_name, seed, device)
                model, reg_head, normalizer = model_info
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
                if normalizer is not None:
                    id_dataset.set_normalizer(normalizer)
                dataloader = PyGDataLoader(id_dataset, batch_size=BATCH_SIZE, shuffle=False)
                predictions, targets, metadata = evaluate_gotennet_detailed(
                    model_info, dataloader, device, target_name, seed
                )
                df = pd.DataFrame({
                    'model': 'gotennet',
                    'property': target_name,
                    'seed': seed,
                    'prediction': predictions,
                    'actual': targets,
                    'temperature': [m.get('temperature') for m in metadata],
                    'composition': [m.get('composition') for m in metadata]
                })
                df.to_csv(combined_path, mode='a', header=first_write, index=False)
                first_write = False
                print(f"Saved results for regression/gotennet/{target_name}/{seed} to {combined_path}")
            except Exception as e:
                print(f"Error evaluating regression/gotennet/{target_name}/{seed}: {e}")
                continue
    print(f"All GotenNet results saved to {combined_path}")

if __name__ == "__main__":
    main() 