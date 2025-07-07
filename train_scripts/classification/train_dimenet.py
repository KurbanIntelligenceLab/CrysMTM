import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

from configs.classifier_config import (
    BASE_DIR,
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_ROTATIONS,
    NUM_EPOCHS,
    SEEDS,
    TRAIN_TEMPS,
)
from dataloaders.classification_dataloader import ClassificationLoader
from models.classification.dimenet_classifier import (
    DEFAULT_MODEL_NAME,
    DimeNetPlusPlusClassifier,
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


def split_dataset(dataset, val_ratio=0.2):
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    split = int(len(indices) * (1 - val_ratio))
    train_idx, val_idx = indices[:split], indices[split:]
    from torch.utils.data import Subset

    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def train(model, loader, optimizer, epoch, num_epochs, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", leave=False)
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch.y.size(0)
        _, predicted = outputs.max(1)
        total += batch.y.size(0)
        correct += predicted.eq(batch.y.view(-1)).sum().item()
        avg_loss = running_loss / total if total > 0 else 0.0
        acc = 100.0 * correct / total if total > 0 else 0.0
        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{acc:.2f}%"})
    avg_loss = running_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc


def evaluate(model, loader, device, desc="Eval"):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    pbar = tqdm(loader, desc=desc, leave=False)
    with torch.no_grad():
        for batch in pbar:
            batch = batch.to(device)
            outputs = model(batch)
            loss = criterion(outputs, batch.y.view(-1))
            running_loss += loss.item() * batch.y.size(0)
            _, predicted = outputs.max(1)
            total += batch.y.size(0)
            correct += predicted.eq(batch.y.view(-1)).sum().item()
            avg_loss = running_loss / total if total > 0 else 0.0
            acc = 100.0 * correct / total if total > 0 else 0.0
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{acc:.2f}%"})
    avg_loss = running_loss / total if total > 0 else 0.0
    acc = 100.0 * correct / total if total > 0 else 0.0
    return avg_loss, acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for seed in SEEDS:
        print(f"\n=== Running for seed {seed} ===")
        set_all_seeds(seed)
        out_dir = os.path.join(
            "results", "classification", DEFAULT_MODEL_NAME, str(seed)
        )
        os.makedirs(out_dir, exist_ok=True)

        # Datasets
        full_train_dataset = ClassificationLoader(
            base_dir=BASE_DIR,
            temperature_filter=lambda temp: temp in TRAIN_TEMPS,
            modalities=["xyz", "element"],
            max_rotations=MAX_ROTATIONS,
            as_pyg_data=True,
        )

        # Split train/val
        train_dataset, val_dataset = split_dataset(full_train_dataset, val_ratio=0.2)
        train_loader = PyGDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = PyGDataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Model
        model = DimeNetPlusPlusClassifier().to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        best_val_acc = 0.0
        best_model_path = os.path.join(out_dir, "best_model.pth")

        # Training loop
        for epoch in range(NUM_EPOCHS):
            train_loss, train_acc = train(
                model, train_loader, optimizer, epoch, NUM_EPOCHS, device
            )
            val_loss, val_acc = evaluate(
                model, val_loader, device, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Val]"
            )
            print(
                f"Seed {seed} | Epoch {epoch + 1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%"
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)


if __name__ == "__main__":
    main()
