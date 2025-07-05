import os
import torch
import torch.nn as nn
import pandas as pd
import pickle
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloaders.regression_dataloader import RegressionLoader
from configs.regression_config import (
    SEEDS, TARGET_PROPERTIES, BASE_DIR, MAX_ROTATIONS, NORMALIZATION_METHOD, ID_TEMPS, BATCH_SIZE, OOD_TEMPS
)
from models.regression.resnet_regressor import DEFAULT_IMAGE_SIZE, DEFAULT_MODEL_NAME, create_resnet_regressor
TEST_TEMPS = ID_TEMPS + OOD_TEMPS
def load_resnet_model(target_name, seed, device):
    """Load trained ResNet model."""
    model_path = os.path.join("results", f"resnet/{target_name}", str(seed), "best_model.pth")
    
    # Load normalizer if available
    normalizer_path = os.path.join("results", f"resnet/{target_name}", str(seed), "normalizer.pkl")
    normalizer = None
    if os.path.exists(normalizer_path):
        with open(normalizer_path, 'rb') as f:
            normalizer = pickle.load(f)
    
    model = create_resnet_regressor(model_name=DEFAULT_MODEL_NAME, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    return model, normalizer

@torch.no_grad()
def collate_fn_resnet(batch):
    """Collate function for ResNet evaluation."""
    # Transforms for images (same as training)
    transform = transforms.Compose([
        transforms.Resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    images = torch.stack([transform(item["image"]) if not isinstance(item["image"], torch.Tensor) else item["image"] for item in batch], dim=0)
    regression_labels = torch.stack([item["regression_label"] for item in batch], dim=0)
    
    # Preserve metadata from the original items
    temperatures = [item.get("temperature", None) for item in batch]
    phases = [item.get("phase", None) for item in batch]
    
    result = {
        "image": images, 
        "regression_label": regression_labels,
        "temperatures": temperatures,
        "phases": phases
    }
    
    return result

def evaluate_resnet_detailed(model_info, dataloader, device, target_name, seed):
    """Evaluate ResNet model and return detailed predictions with metadata."""
    model, normalizer = model_info
    model.eval()
    
    predictions = []
    targets = []
    metadata = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # ResNet specific logic - uses only images
            images = batch["image"].to(device)
            outputs = model(images)
            
            # Get target for the specific property
            target_index = TARGET_PROPERTIES.index(target_name)
            labels = batch["regression_label"]
            if labels.dim() == 2:
                target = labels[:, target_index].unsqueeze(1)
            else:
                target = labels.unsqueeze(1)
            
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

            # Extract metadata - one entry per sample in the batch
            batch_size = len(batch['temperatures'])
            # Only use the first batch_size predictions/targets
            predictions.extend(pred_batch.flatten()[:batch_size])
            targets.extend(target_batch.flatten()[:batch_size])

            for i in range(batch_size):
                temperature = batch["temperatures"][i]
                composition = batch["phases"][i]
                if temperature is None or composition is None:
                    print(f"WARNING: Missing metadata at batch {batch_idx}, sample {i}")
                metadata.append({
                    'model': 'resnet',
                    'target': target_name,
                    'seed': seed,
                    'batch_idx': i,
                    'sample_idx': batch_idx * BATCH_SIZE + i,
                    'temperature': temperature,
                    'composition': composition
                })
    
    return predictions, targets, metadata

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    output_dir = "results/detailed_analysis"
    os.makedirs(output_dir, exist_ok=True)
    combined_path = os.path.join(output_dir, "resnet_predictions.csv")
    first_write = not os.path.exists(combined_path)
    
    for target_name in TARGET_PROPERTIES:
        for seed in SEEDS:
            try:
                model_path = os.path.join("results", f"resnet/{target_name}", str(seed), "best_model.pth")
                if not os.path.exists(model_path):
                    print(f"Model not found: {model_path}")
                    continue
                
                model_info = load_resnet_model(target_name, seed, device)
                model, normalizer = model_info
                
                # Use appropriate dataloader logic for ResNet
                id_dataset = RegressionLoader(
                    label_dir=BASE_DIR,
                    temperature_filter=lambda temp: temp in TEST_TEMPS,
                    modalities=["image"],
                    transform=None,  # We'll apply transforms in collate_fn
                    max_rotations=MAX_ROTATIONS,
                    normalize_labels=False,  # Don't normalize in dataset, handle it manually
                    normalization_method=NORMALIZATION_METHOD,
                    fit_normalizer_on_data=False,
                )
                
                # Use the same collate function as training
                dataloader = DataLoader(
                    id_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                    num_workers=0, collate_fn=collate_fn_resnet
                )
                
                predictions, targets, metadata = evaluate_resnet_detailed(
                    model_info, dataloader, device, target_name, seed
                )
                
                df = pd.DataFrame({
                    'model': 'resnet',
                    'target': target_name,
                    'seed': seed,
                    'prediction': predictions,
                    'target': targets,
                    'temperature': [m.get('temperature') for m in metadata],
                    'composition': [m.get('composition') for m in metadata]
                })
                
                df.to_csv(combined_path, mode='a', header=first_write, index=False)
                first_write = False
                print(f"Saved results for resnet/{target_name}/{seed} to {combined_path}")
                
            except Exception as e:
                print(f"Error evaluating resnet/{target_name}/{seed}: {e}")
                continue
    
    print(f"All ResNet results saved to {combined_path}")

if __name__ == "__main__":
    main() 