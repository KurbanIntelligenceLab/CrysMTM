import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import global_mean_pool, knn_graph
from tqdm import tqdm

from configs.regression_config import (
    BASE_DIR,
    BATCH_SIZE,
    ID_TEMPS,
    LEARNING_RATE,
    MAX_ROTATIONS,
    NUM_EPOCHS,
    OOD_TEMPS,
    SEEDS,
)
from dataloaders.regression_dataloader import RegressionLoader
from models.regression.egnn_regressor import EGNNRegressor

TARGET_NAMES = ["HOMO", "LUMO", "Eg", "Ef", "Et"]


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
    from torch.utils.data import Subset
    return Subset(dataset, train_idx), Subset(dataset, val_idx)

def train(model, reg_head, loader, optimizer, epoch, num_epochs, device, target_index):
    model.train()
    reg_head.train()
    running_loss = 0.0
    total = 0
    criterion = nn.MSELoss()
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
    for batch in pbar:
        batch = batch.to(device)
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

def evaluate(model, reg_head, loader, device, desc, target_index):
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
            labels = batch.y
            if labels.dim() == 1:
                target = labels.unsqueeze(1)
            else:
                target = labels[:, target_index].unsqueeze(1)
            loss = criterion(outputs, target)
            running_loss += loss.item() * target.size(0)
            mae = torch.mean(torch.abs(outputs - target)).item()
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
    for target_index, target_name in enumerate(TARGET_NAMES):
        print(f"\n=== Running for target {target_name} ===")
        for seed in SEEDS:
            print(f"\n=== Running for seed {seed} ===")
            set_all_seeds(seed)
            out_dir = os.path.join(
                "results", "regression", f"egnn_{target_name}", str(seed)
            )
            os.makedirs(out_dir, exist_ok=True)

            # Datasets
            full_train_dataset = RegressionLoader(
                label_dir=BASE_DIR,
                temperature_filter=lambda temp: (0 <= temp <= 800)
                and (temp not in ID_TEMPS),
                modalities=["xyz", "element"],
                max_rotations=MAX_ROTATIONS,
                as_pyg_data=True,
            )
            id_dataset = RegressionLoader(
                label_dir=BASE_DIR,
                temperature_filter=lambda temp: temp in ID_TEMPS,
                modalities=["xyz", "element"],
                max_rotations=MAX_ROTATIONS,
                as_pyg_data=True,
            )
            ood_dataset = RegressionLoader(
                label_dir=BASE_DIR,
                temperature_filter=lambda temp: temp in OOD_TEMPS,
                modalities=["xyz", "element"],
                max_rotations=MAX_ROTATIONS,
                as_pyg_data=True,
            )

            # Split train/val
            train_dataset, val_dataset = split_dataset(
                full_train_dataset, val_ratio=0.2, seed=seed
            )
            train_loader = PyGDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = PyGDataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
            id_loader = PyGDataLoader(id_dataset, batch_size=BATCH_SIZE, shuffle=False)
            ood_loader = PyGDataLoader(ood_dataset, batch_size=BATCH_SIZE, shuffle=False)

            # Model
            model = EGNNRegressor.to(device)
            reg_head = nn.Linear(1, 1).to(device)
            optimizer = optim.Adam(list(model.parameters()) + list(reg_head.parameters()), lr=LEARNING_RATE)
            best_val_mae = float('inf')
            best_model_path = os.path.join(out_dir, "best_model.pth")
            best_reg_head_path = os.path.join(out_dir, "best_reg_head.pth")

            # Training loop
            for epoch in range(NUM_EPOCHS):
                train_loss = train(
                    model, reg_head, train_loader, optimizer, epoch, NUM_EPOCHS, device, target_index
                )
                val_loss, val_mae = evaluate(
                    model, reg_head, val_loader, device, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]", target_index=target_index
                )
                print(
                    f"Target {target_name} | Seed {seed} | Epoch {epoch+1}/{NUM_EPOCHS} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f} | Val MAE: {val_mae:.4f}"
                )
                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    torch.save(model.state_dict(), best_model_path)
                    torch.save(reg_head.state_dict(), best_reg_head_path)

            # Load best model and evaluate on ID/OOD
            model.load_state_dict(torch.load(best_model_path))
            reg_head.load_state_dict(torch.load(best_reg_head_path))
            _, id_mae = evaluate(model, reg_head, id_loader, device, desc="ID Test", target_index=target_index)
            _, ood_mae = evaluate(model, reg_head, ood_loader, device, desc="OOD Test", target_index=target_index)
            print(
                f"Target {target_name} | Seed {seed} | ID Test MAE: {id_mae:.4f} | OOD Test MAE: {ood_mae:.4f}"
            )

            with open(os.path.join(out_dir, "results.txt"), "w") as f:
                f.write(f"Best Val MAE: {best_val_mae:.4f}\n")
                f.write(f"ID Test MAE: {id_mae:.4f}\n")
                f.write(f"OOD Test MAE: {ood_mae:.4f}\n")
            print(f"Results and best model saved in {out_dir}")

# # Utility: scan for max number of nodes in any graph
# if __name__ == "__main__":
#     dataset = RegressionLoader(
#         label_dir=BASE_DIR,
#         modalities=["xyz", "element"],
#         as_pyg_data=True,
#     )
#     max_nodes = max([data.z.size(0) for data in dataset])
#     print(f"Max nodes in any graph: {max_nodes}")
#     exit(0)

if __name__ == "__main__":
    main() 