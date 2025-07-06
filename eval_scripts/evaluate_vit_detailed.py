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
from models.regression.vit_regressor import DEFAULT_IMAGE_SIZE, DEFAULT_MODEL_NAME, create_vit_regressor
TEST_TEMPS = ID_TEMPS + OOD_TEMPS
def load_vit_model(target_name, seed, device):
    """Load trained ViT model."""
    model_path = os.path.join("results", f"regression/vit/{target_name}", str(seed), "best_model.pth")
    
    # Load normalizer if available
    normalizer_path = os.path.join("results", f"regression/vit/{target_name}", str(seed), "normalizer.pkl")
    normalizer = None
    if os.path.exists(normalizer_path):
        with open(normalizer_path, 'rb') as f:
            normalizer = pickle.load(f)
    
    model = create_vit_regressor(model_name=DEFAULT_MODEL_NAME, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    return model, normalizer

@torch.no_grad()
def collate_fn_vit(batch):
    """Collate function for ViT evaluation."""
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

def evaluate_vit_detailed(model_info, dataloader, device, target_name, seed):
    """Evaluate ViT model and return detailed predictions with metadata."""
    model, normalizer = model_info
    model.eval()
    
    predictions = []
    targets = []
    metadata = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # ViT specific logic - uses only images
            images = batch["image"].to(device)
            outputs = model(images)
            
            # Get target for the specific property
            target_index = TARGET_PROPERTIES.index(target_name)
            labels = batch["regression_label"]
            if labels.dim() == 2:
                target = labels[:, target_index].unsqueeze(1)
            else:
                target = labels.unsqueeze(1)
            
            # Keep both predictions and targets in normalized scale for fair comparison
            # The model outputs are already in normalized scale
            # The targets from the dataloader are also in normalized scale
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
                    'model': 'vit',
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
    combined_path = os.path.join(output_dir, "vit_predictions.csv")
    first_write = not os.path.exists(combined_path)
    
    for target_name in TARGET_PROPERTIES:
        for seed in SEEDS:
            try:
                model_path = os.path.join("results", f"regression/vit/{target_name}", str(seed), "best_model.pth")
                if not os.path.exists(model_path):
                    print(f"Model not found: {model_path}")
                    continue
                
                model_info = load_vit_model(target_name, seed, device)
                model, normalizer = model_info
                
                # Use appropriate dataloader logic for ViT
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
                    num_workers=0, collate_fn=collate_fn_vit
                )
                
                predictions, targets, metadata = evaluate_vit_detailed(
                    model_info, dataloader, device, target_name, seed
                )
                
                df = pd.DataFrame({
                    'model': 'vit',
                    'property': target_name,
                    'seed': seed,
                    'prediction': predictions,
                    'actual': targets,
                    'temperature': [m.get('temperature') for m in metadata],
                    'composition': [m.get('composition') for m in metadata]
                })
                
                df.to_csv(combined_path, mode='a', header=first_write, index=False)
                first_write = False
                print(f"Saved results for regression/vit/{target_name}/{seed} to {combined_path}")
                
            except Exception as e:
                print(f"Error evaluating regression/vit/{target_name}/{seed}: {e}")
                continue
    
    print(f"All ViT results saved to {combined_path}")

if __name__ == "__main__":
    main() 