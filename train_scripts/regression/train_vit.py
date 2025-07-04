import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from configs.regression_config import (
    BASE_DIR,
    BATCH_SIZE,
    EarlyStopping,
    ID_TEMPS,
    LEARNING_RATE,
    MAX_ROTATIONS,
    NUM_EPOCHS,
    NUM_WORKERS,
    OOD_TEMPS,
    SEEDS,
    TARGET_PROPERTIES,
    TRAIN_TEMPS,
    NORMALIZE_LABELS,
    NORMALIZATION_METHOD,
)
from dataloaders.regression_dataloader import RegressionLoader
from models.regression.vit_regressor import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MODEL_NAME,
    create_vit_regressor,
)


# Transforms for images
transform = transforms.Compose(
    [
        transforms.Resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def split_dataset(dataset, val_ratio=0.2, seed=42):
    np.random.seed(seed)
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    split = int(len(indices) * (1 - val_ratio))
    train_idx, val_idx = indices[:split], indices[split:]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)

def train(model, loader, optimizer, epoch, num_epochs, device, target_index):
    model.train()
    running_loss = 0.0
    total = 0
    criterion = nn.MSELoss()
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        labels = batch["regression_label"]
        if labels.dim() == 2:
            target = labels[:, target_index].unsqueeze(1).to(device)
        else:
            target = labels.unsqueeze(1).to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * target.size(0)
        total += target.size(0)
        avg_loss = running_loss / total if total > 0 else 0.0
        pbar.set_postfix({"mse": f"{avg_loss:.4f}"})
    avg_loss = running_loss / total
    return avg_loss

def evaluate(model, loader, device, desc, target_index, normalizer=None):
    model.eval()
    total = 0
    running_loss = 0.0
    running_mae = 0.0
    criterion = nn.MSELoss()
    pbar = tqdm(loader, desc=desc, leave=False)
    with torch.no_grad():
        for batch in pbar:
            images = batch["image"].to(device)
            labels = batch["regression_label"]
            if labels.dim() == 2:
                target = labels[:, target_index].unsqueeze(1).to(device)
            else:
                target = labels.unsqueeze(1).to(device)
            outputs = model(images)
            loss = criterion(outputs, target)
            
            # Convert to original scale for MAE calculation if normalizer is used
            if normalizer is not None:
                # Convert normalized predictions and targets back to original scale
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
                mae = torch.mean(torch.abs(outputs_orig - target_orig)).item()
            else:
                mae = torch.mean(torch.abs(outputs - target)).item()
            
            running_loss += loss.item() * target.size(0)
            running_mae += mae * target.size(0)
            total += target.size(0)
            avg_loss = running_loss / total if total > 0 else 0.0
            avg_mae = running_mae / total if total > 0 else 0.0
            pbar.set_postfix({"mse": f"{avg_loss:.4f}", "mae": f"{avg_mae:.4f}"})
    avg_loss = running_loss / total if total > 0 else 0.0
    avg_mae = running_mae / total if total > 0 else 0.0
    return avg_loss, avg_mae

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for target_index, target_name in enumerate(TARGET_PROPERTIES):
        print(f"\n=== Running for target {target_name} ===")
        for seed in SEEDS:
            print(f"\n=== Running for seed {seed} ===")
            set_all_seeds(seed)
            out_dir = os.path.join(
                "results", "regression", f"vit/{target_name}", str(seed)
            )
            os.makedirs(out_dir, exist_ok=True)

            # Datasets
            full_train_dataset = RegressionLoader(
                label_dir=BASE_DIR,
                temperature_filter=lambda temp: temp in TRAIN_TEMPS,
                modalities=["image"],
                transform=transform,
                max_rotations=MAX_ROTATIONS,
                normalize_labels=NORMALIZE_LABELS,
                normalization_method=NORMALIZATION_METHOD,
                fit_normalizer_on_data=True,  # Fit on training data
            )
            
            # Get the fitted normalizer from training dataset
            normalizer = full_train_dataset.normalizer if NORMALIZE_LABELS else None
            
            id_dataset = RegressionLoader(
                label_dir=BASE_DIR,
                temperature_filter=lambda temp: temp in ID_TEMPS,
                modalities=["image"],
                transform=transform,
                max_rotations=MAX_ROTATIONS,
                normalize_labels=NORMALIZE_LABELS,
                normalization_method=NORMALIZATION_METHOD,
                fit_normalizer_on_data=False,
            )
            ood_dataset = RegressionLoader(
                label_dir=BASE_DIR,
                temperature_filter=lambda temp: temp in OOD_TEMPS,
                modalities=["image"],
                transform=transform,
                max_rotations=MAX_ROTATIONS,
                normalize_labels=NORMALIZE_LABELS,
                normalization_method=NORMALIZATION_METHOD,
                fit_normalizer_on_data=False,
            )
            
            # Set the fitted normalizer for ID and OOD datasets
            if normalizer is not None:
                id_dataset.set_normalizer(normalizer)
                ood_dataset.set_normalizer(normalizer)

            # Split train/val
            train_dataset, val_dataset = split_dataset(
                full_train_dataset, val_ratio=0.2, seed=seed
            )
            train_loader = DataLoader(
                train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
            )
            val_loader = DataLoader(
                val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
            )
            id_loader = DataLoader(
                id_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
            )
            ood_loader = DataLoader(
                ood_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
            )

            # Model (ViT, regression head)
            model = create_vit_regressor(
                model_name=DEFAULT_MODEL_NAME, pretrained=False
            ).to(device)
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            best_val_mae = float('inf')
            best_model_path = os.path.join(out_dir, "best_model.pth")
            early_stopping = EarlyStopping()

            # Training loop
            for epoch in range(NUM_EPOCHS):
                train_loss = train(
                    model, train_loader, optimizer, epoch, NUM_EPOCHS, device, target_index
                )
                val_loss, val_mae = evaluate(
                    model, val_loader, device, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]", target_index=target_index, normalizer=normalizer
                )
                print(
                    f"Target {target_name} | Seed {seed} | Epoch {epoch+1}/{NUM_EPOCHS} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f} | Val MAE: {val_mae:.4f}"
                )
                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    torch.save(model.state_dict(), best_model_path)
                
                # Early stopping check
                if early_stopping(val_mae):
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

            # Load best model and evaluate on ID/OOD
            model.load_state_dict(torch.load(best_model_path))
            id_loss, id_mae = evaluate(model, id_loader, device, desc="ID Test", target_index=target_index, normalizer=normalizer)
            ood_loss, ood_mae = evaluate(model, ood_loader, device, desc="OOD Test", target_index=target_index, normalizer=normalizer)
            print(
                f"Target {target_name} | Seed {seed} | ID Test MAE: {id_mae:.4f} | OOD Test MAE: {ood_mae:.4f}"
            )

            with open(os.path.join(out_dir, "results.txt"), "w") as f:
                f.write(f"Best Val MAE: {best_val_mae:.4f}\n")
                f.write(f"ID Test MSE: {id_loss:.4f}\n")
                f.write(f"OOD Test MSE: {ood_loss:.4f}\n")
                f.write(f"ID Test MAE: {id_mae:.4f}\n")
                f.write(f"OOD Test MAE: {ood_mae:.4f}\n")
            
            # Save normalizer if used
            if normalizer is not None:
                import pickle
                with open(os.path.join(out_dir, "normalizer.pkl"), "wb") as f:
                    pickle.dump(normalizer, f)
                print(f"Normalizer saved to {os.path.join(out_dir, 'normalizer.pkl')}")
            
            print(f"Results and best model saved in {out_dir}")

if __name__ == "__main__":
    main() 