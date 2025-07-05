import os
import torch
import torch.nn as nn
import pandas as pd
import pickle
from torch_geometric.loader import DataLoader as PyGDataLoader

from dataloaders.regression_dataloader import RegressionLoader
from configs.regression_config import (
    SEEDS, TARGET_PROPERTIES, BASE_DIR, MAX_ROTATIONS, NORMALIZATION_METHOD, ID_TEMPS, BATCH_SIZE, OOD_TEMPS
)
from models.regression.visnet_regressor import ViSNetRegressor
TEST_TEMPS = ID_TEMPS + OOD_TEMPS
def load_visnet_model(target_name, seed, device):
    """Load trained ViSNet model."""
    model_path = os.path.join("results", f"regression/visnet/{target_name}", str(seed), "best_model.pth")
    
    # Load normalizer if available
    normalizer_path = os.path.join("results", f"regression/visnet/{target_name}", str(seed), "normalizer.pkl")
    normalizer = None
    if os.path.exists(normalizer_path):
        with open(normalizer_path, 'rb') as f:
            normalizer = pickle.load(f)
    
    model = ViSNetRegressor().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    return model, normalizer

def evaluate_visnet_detailed(model_info, dataloader, device, target_name, seed):
    """Evaluate ViSNet model and return detailed predictions with metadata."""
    model, normalizer = model_info
    model.eval()
    
    predictions = []
    targets = []
    metadata = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device)
            
            # ViSNet specific logic - uses entire batch object
            outputs = model(batch)
            
            # Get target for the specific property
            target_index = TARGET_PROPERTIES.index(target_name)
            labels = batch.regression_label if hasattr(batch, 'regression_label') else batch.y
            if labels.dim() == 1:
                target = labels.unsqueeze(1)
            else:
                target = labels[:, target_index].unsqueeze(1)
            
            # Apply normalization manually if needed
            if normalizer is not None:
                target = torch.tensor(
                    normalizer.transform(target.cpu().numpy()),
                    dtype=torch.float,
                    device=device
                )
            
            # Convert to original scale if normalizer is used
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

            # Extract metadata from PyG batch
            batch_size = batch.num_graphs
            # Only use the first batch_size predictions/targets
            predictions.extend(pred_batch.flatten()[:batch_size])
            targets.extend(target_batch.flatten()[:batch_size])

            # For PyG data, we need to extract metadata from the dataset
            # Get the dataset indices for this batch
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + batch_size, len(dataloader.dataset))
            
            for i in range(batch_size):
                sample_idx = start_idx + i
                if sample_idx < len(dataloader.dataset):
                    # Get metadata from the dataset - handle PyG data properly
                    try:
                        sample = dataloader.dataset[sample_idx]
                        # Try different ways to access metadata
                        if hasattr(sample, 'temperature'):
                            temperature = sample.temperature
                        elif hasattr(sample, 'metadata') and hasattr(sample.metadata, 'temperature'):
                            temperature = sample.metadata.temperature
                        elif hasattr(dataloader.dataset, 'data') and sample_idx < len(dataloader.dataset.data):
                            # Try to access from dataset.data if available
                            dataset_entry = dataloader.dataset.data[sample_idx]
                            if isinstance(dataset_entry, dict):
                                temperature = dataset_entry.get('temperature', None)
                            else:
                                temperature = None
                        else:
                            temperature = None
                        
                        if hasattr(sample, 'phase'):
                            composition = sample.phase
                        elif hasattr(sample, 'metadata') and hasattr(sample.metadata, 'phase'):
                            composition = sample.metadata.phase
                        elif hasattr(dataloader.dataset, 'data') and sample_idx < len(dataloader.dataset.data):
                            # Try to access from dataset.data if available
                            dataset_entry = dataloader.dataset.data[sample_idx]
                            if isinstance(dataset_entry, dict):
                                composition = dataset_entry.get('phase', None)
                            else:
                                composition = None
                        else:
                            composition = None
                            
                    except Exception as e:
                        temperature = None
                        composition = None
                        print(f"WARNING: Error extracting metadata at batch {batch_idx}, sample {i}: {e}")
                    
                    if temperature is None or composition is None:
                        print(f"WARNING: Missing metadata at batch {batch_idx}, sample {i}")
                    
                    metadata.append({
                        'model': 'visnet',
                        'target': target_name,
                        'seed': seed,
                        'batch_idx': i,
                        'sample_idx': sample_idx,
                        'temperature': temperature,
                        'composition': composition
                    })

    return predictions, targets, metadata

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    output_dir = "results/detailed_analysis"
    os.makedirs(output_dir, exist_ok=True)
    combined_path = os.path.join(output_dir, "visnet_predictions.csv")
    first_write = not os.path.exists(combined_path)
    
    for target_name in TARGET_PROPERTIES:
        for seed in SEEDS:
            try:
                model_path = os.path.join("results", f"regression/visnet/{target_name}", str(seed), "best_model.pth")
                if not os.path.exists(model_path):
                    print(f"Model not found: {model_path}")
                    continue
                
                model_info = load_visnet_model(target_name, seed, device)
                model, normalizer = model_info
                
                # Use appropriate dataloader logic for ViSNet (PyG data)
                id_dataset = RegressionLoader(
                    label_dir=BASE_DIR,
                    temperature_filter=lambda temp: temp in TEST_TEMPS,
                    modalities=["xyz", "element"],
                    max_rotations=MAX_ROTATIONS,
                    as_pyg_data=True,
                    normalize_labels=False,  # Don't normalize in dataset, handle it manually
                    normalization_method=NORMALIZATION_METHOD,
                    fit_normalizer_on_data=False,
                )
                
                # Use PyG DataLoader
                dataloader = PyGDataLoader(
                    id_dataset, batch_size=BATCH_SIZE, shuffle=False
                )
                
                predictions, targets, metadata = evaluate_visnet_detailed(
                    model_info, dataloader, device, target_name, seed
                )
                
                df = pd.DataFrame({
                    'model': 'visnet',
                    'target': target_name,
                    'seed': seed,
                    'prediction': predictions,
                    'target': targets,
                    'temperature': [m.get('temperature') for m in metadata],
                    'composition': [m.get('composition') for m in metadata]
                })
                
                df.to_csv(combined_path, mode='a', header=first_write, index=False)
                first_write = False
                print(f"Saved results for regression/visnet/{target_name}/{seed} to {combined_path}")
                
            except Exception as e:
                print(f"Error evaluating regression/visnet/{target_name}/{seed}: {e}")
                continue
    
    print(f"All ViSNet results saved to {combined_path}")

if __name__ == "__main__":
    main() 