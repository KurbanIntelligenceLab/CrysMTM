import os
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from configs.classifier_config import (
    BASE_DIR,
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_ROTATIONS,
    NUM_EPOCHS,
    NUM_WORKERS,
    SEEDS,
    TRAIN_TEMPS,
)
from dataloaders.classification_dataloader import ClassificationLoader
from models.classification.bert_classifier import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_NAME,
    create_bert_classifier,
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
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def collate_fn_bert(batch, tokenizer):
    texts = [item["text"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    encodings = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=DEFAULT_MAX_LENGTH,
    )
    return {**encodings, "labels": labels}


def train(model, loader, optimizer, epoch, num_epochs, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", leave=False)
    for batch in pbar:
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        optimizer.zero_grad()
        outputs = model(batch)
        loss = outputs.loss
        logits = outputs.logits
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch["labels"].size(0)
        _, predicted = logits.max(1)
        total += batch["labels"].size(0)
        correct += predicted.eq(batch["labels"]).sum().item()
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
    pbar = tqdm(loader, desc=desc, leave=False)
    with torch.no_grad():
        for batch in pbar:
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            outputs = model(batch)
            loss = outputs.loss
            logits = outputs.logits
            running_loss += loss.item() * batch["labels"].size(0)
            _, predicted = logits.max(1)
            total += batch["labels"].size(0)
            correct += predicted.eq(batch["labels"]).sum().item()
            avg_loss = running_loss / total if total > 0 else 0.0
            acc = 100.0 * correct / total if total > 0 else 0.0
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{acc:.2f}%"})
    avg_loss = running_loss / total if total > 0 else 0.0
    acc = 100.0 * correct / total if total > 0 else 0.0
    return avg_loss, acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model and get tokenizer
    model = create_bert_classifier(
        model_name=DEFAULT_MODEL_NAME, num_classes=3, max_length=DEFAULT_MAX_LENGTH
    ).to(device)
    tokenizer = model.get_tokenizer()

    for seed in SEEDS:
        print(f"\n=== Running for seed {seed} ===")
        set_all_seeds(seed)
        # Prepare output directory
        model_name = "bert"
        out_dir = os.path.join("results", "classification", model_name, str(seed))
        os.makedirs(out_dir, exist_ok=True)

        # Datasets
        full_train_dataset = ClassificationLoader(
            base_dir=BASE_DIR,
            temperature_filter=lambda temp: temp in TRAIN_TEMPS,
            modalities=["text"],
            max_rotations=MAX_ROTATIONS,
        )

        train_dataset, val_dataset = split_dataset(full_train_dataset, val_ratio=0.2)
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            collate_fn=lambda b: collate_fn_bert(b, tokenizer),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=lambda b: collate_fn_bert(b, tokenizer),
        )

        # Optimizer
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
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

        print(f"Best model saved in {out_dir}")


if __name__ == "__main__":
    main()
