import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm

from configs.regression_config import (
    BASE_DIR,
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_ROTATIONS,
    NORMALIZATION_METHOD,
    NORMALIZE_LABELS,
    NUM_EPOCHS,
    SEEDS,
    TARGET_PROPERTIES,
    TRAIN_TEMPS,
    EarlyStopping,
)
from dataloaders.regression_dataloader import RegressionLoader
from models.regression.gotennet_regressor import GotenNetRegressor


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_dataset(dataset, val_ratio=0.2):
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    split = int(len(indices) * (1 - val_ratio))
    train_idx, val_idx = indices[:split], indices[split:]
    from torch.utils.data import Subset

    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def train(model, reg_head, loader, optimizer, epoch, num_epochs, device, target_index):
    model.train()
    reg_head.train()
    running_loss = 0.0
    total = 0
    criterion = nn.MSELoss()
    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", leave=False)
    for batch in pbar:
        batch = batch.to(device)
        h, X = model(batch)  # h: [num_nodes, hidden_dim]
        pooled = global_mean_pool(h, batch.batch)
        outputs = reg_head(pooled)
        labels = batch.y
        if labels.dim() == 1:
            target = labels.unsqueeze(1)
        else:
            target = labels[:, target_index].unsqueeze(1)
        loss = criterion(outputs, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(reg_head.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item() * target.size(0)
        total += target.size(0)
        avg_loss = running_loss / total if total > 0 else 0.0
        pbar.set_postfix({"mse": f"{avg_loss:.4f}"})
    avg_loss = running_loss / total
    return avg_loss


def evaluate(model, reg_head, loader, device, desc, target_index, normalizer=None):
    model.eval()
    reg_head.eval()
    total = 0
    running_loss = 0.0
    running_mae = 0.0
    criterion = nn.MSELoss()
    pbar = tqdm(loader, desc=desc, leave=False)
    with torch.no_grad():
        for batch in pbar:
            batch = batch.to(device)
            h, X = model(batch)
            pooled = global_mean_pool(h, batch.batch)
            outputs = reg_head(pooled)
            labels = batch.y
            if labels.dim() == 1:
                target = labels.unsqueeze(1)
            else:
                target = labels[:, target_index].unsqueeze(1)
            loss = criterion(outputs, target)

            # Convert to original scale for MAE calculation if normalizer is used
            if normalizer is not None:
                # Convert normalized predictions and targets back to original scale
                outputs_orig = torch.tensor(
                    normalizer.inverse_transform(outputs.cpu().numpy()),
                    dtype=torch.float,
                    device=device,
                )
                target_orig = torch.tensor(
                    normalizer.inverse_transform(target.cpu().numpy()),
                    dtype=torch.float,
                    device=device,
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
                "results", "regression", f"gotennet/{target_name}", str(seed)
            )
            os.makedirs(out_dir, exist_ok=True)

            # Datasets
            full_train_dataset = RegressionLoader(
                label_dir=BASE_DIR,
                temperature_filter=lambda temp: temp in TRAIN_TEMPS,
                modalities=["xyz", "element"],
                max_rotations=MAX_ROTATIONS,
                as_pyg_data=True,
                normalize_labels=NORMALIZE_LABELS,
                normalization_method=NORMALIZATION_METHOD,
                fit_normalizer_on_data=True,  # Fit on training data
            )

            # Get the fitted normalizer from training dataset
            normalizer = full_train_dataset.normalizer if NORMALIZE_LABELS else None

            # Split train/val
            train_dataset, val_dataset = split_dataset(
                full_train_dataset, val_ratio=0.2
            )
            train_loader = PyGDataLoader(
                train_dataset, batch_size=BATCH_SIZE, shuffle=True
            )
            val_loader = PyGDataLoader(
                val_dataset, batch_size=BATCH_SIZE, shuffle=False
            )

            # Model
            model = GotenNetRegressor.to(device)
            # Determine hidden dim from model (assume n_atom_basis)
            hidden_dim = 32  # n_atom_basis
            reg_head = nn.Linear(hidden_dim, 1).to(device)
            optimizer = optim.Adam(
                list(model.parameters()) + list(reg_head.parameters()), lr=LEARNING_RATE
            )
            best_val_mae = float("inf")
            best_model_path = os.path.join(out_dir, "best_model.pth")
            best_reg_head_path = os.path.join(out_dir, "best_reg_head.pth")
            early_stopping = EarlyStopping()

            # Training loop
            for epoch in range(NUM_EPOCHS):
                train_loss = train(
                    model,
                    reg_head,
                    train_loader,
                    optimizer,
                    epoch,
                    NUM_EPOCHS,
                    device,
                    target_index,
                )
                val_loss, val_mae = evaluate(
                    model,
                    reg_head,
                    val_loader,
                    device,
                    desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Val]",
                    target_index=target_index,
                    normalizer=normalizer,
                )
                print(
                    f"Target {target_name} | Seed {seed} | Epoch {epoch + 1}/{NUM_EPOCHS} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f} | Val MAE: {val_mae:.4f}"
                )
                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    torch.save(model.state_dict(), best_model_path)
                    torch.save(reg_head.state_dict(), best_reg_head_path)

                # Early stopping check
                if early_stopping(val_mae):
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break

            # Save normalizer if used
            if normalizer is not None:
                import pickle

                with open(os.path.join(out_dir, "normalizer.pkl"), "wb") as f:
                    pickle.dump(normalizer, f)
                print(f"Normalizer saved to {os.path.join(out_dir, 'normalizer.pkl')}")

            print(f"Results and best model saved in {out_dir}")


if __name__ == "__main__":
    main()
