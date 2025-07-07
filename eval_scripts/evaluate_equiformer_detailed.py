import os
import pickle

import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader as PyGDataLoader

from configs.regression_config import (
    BASE_DIR,
    BATCH_SIZE,
    ID_TEMPS,
    MAX_ROTATIONS,
    NORMALIZATION_METHOD,
    NORMALIZE_LABELS,
    OOD_TEMPS,
    SEEDS,
    TARGET_PROPERTIES,
)
from dataloaders.regression_dataloader import RegressionLoader

TEST_TEMPS = ID_TEMPS + OOD_TEMPS


def load_equiformer_model(target_name, seed, device):
    """Load trained Equiformer model."""
    model_path = os.path.join(
        "results", f"regression/equiformer/{target_name}", str(seed), "best_model.pth"
    )

    # Load normalizer if available
    normalizer_path = os.path.join(
        "results", f"regression/equiformer/{target_name}", str(seed), "normalizer.pkl"
    )
    normalizer = None
    if os.path.exists(normalizer_path):
        with open(normalizer_path, "rb") as f:
            normalizer = pickle.load(f)

    from models.regression.equiformer_regressor import EquiformerRegressor

    model = EquiformerRegressor().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model, normalizer


def add_node_atom(batch):
    """Add node_atom attribute (atomic numbers) if not present - required for Equiformer."""
    if not hasattr(batch, "node_atom"):
        # Try to infer from element symbols if available
        if hasattr(batch, "element"):
            element_to_z = {"Ti": 22, "O": 8}
            batch.node_atom = torch.tensor(
                [element_to_z.get(e, 0) for e in batch.element],
                dtype=torch.long,
                device=batch.pos.device,
            )
        elif hasattr(batch, "z"):
            batch.node_atom = batch.z
        elif hasattr(batch, "atomic_numbers"):
            batch.node_atom = batch.atomic_numbers
        else:
            raise ValueError(
                "Cannot infer node_atom (atomic numbers) for Equiformer input."
            )
    return batch


def evaluate_equiformer_detailed(model_info, dataloader, device, target_name, seed):
    """Evaluate Equiformer model and return detailed predictions with metadata."""
    model, normalizer = model_info
    model.eval()

    predictions = []
    targets = []
    metadata = []

    # Track the actual dataset indices
    dataset_idx_counter = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device)

            # Equiformer specific preprocessing
            batch = add_node_atom(batch)
            outputs = model(batch)

            # Get target for the specific property
            target_index = TARGET_PROPERTIES.index(target_name)
            labels = batch.y
            if labels.dim() == 1:
                target = labels.unsqueeze(1)
            else:
                target = labels[:, target_index].unsqueeze(1)

            # Keep both predictions and targets in normalized scale for fair comparison
            # The model outputs are already in normalized scale
            # The targets from the dataloader are also in normalized scale
            pred_batch = outputs.cpu().numpy()
            target_batch = target.cpu().numpy()

            # Extract metadata - one entry per sample in the batch
            # Use batch['temperature'] if available for batch size
            if hasattr(batch, "temperature") or hasattr(batch, "temperatures"):
                batch_size = len(
                    getattr(batch, "temperature", getattr(batch, "temperatures", []))
                )
            elif hasattr(batch, "batch") and batch.batch is not None:
                batch_size = batch.batch.max().item() + 1
            else:
                batch_size = len(target_batch.flatten())
            # Only use the first batch_size predictions/targets
            predictions.extend(pred_batch.flatten()[:batch_size])
            targets.extend(target_batch.flatten()[:batch_size])

            for i in range(batch_size):
                # Try to get metadata from dataset
                dataset_idx = dataset_idx_counter + i
                if hasattr(dataloader.dataset, "data") and dataset_idx < len(
                    dataloader.dataset.data
                ):
                    entry = dataloader.dataset.data[dataset_idx]
                    temperature = entry.get("temperature", None)
                    composition = entry.get("phase", None)
                else:
                    temperature = None
                    composition = None
                if temperature is None or composition is None:
                    print(f"WARNING: Missing metadata at batch {batch_idx}, sample {i}")
                metadata.append(
                    {
                        "model": "equiformer",
                        "target": target_name,
                        "seed": seed,
                        "batch_idx": i,
                        "sample_idx": dataset_idx,
                        "temperature": temperature,
                        "composition": composition,
                    }
                )
            # Update the counter for the next batch
            dataset_idx_counter += batch_size

    return predictions, targets, metadata


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    output_dir = "results/detailed_analysis"
    os.makedirs(output_dir, exist_ok=True)
    combined_path = os.path.join(output_dir, "equiformer_predictions.csv")
    first_write = not os.path.exists(combined_path)

    for target_name in TARGET_PROPERTIES:
        for seed in SEEDS:
            try:
                model_path = os.path.join(
                    "results",
                    f"regression/equiformer/{target_name}",
                    str(seed),
                    "best_model.pth",
                )
                if not os.path.exists(model_path):
                    print(f"Model not found: {model_path}")
                    continue

                model_info = load_equiformer_model(target_name, seed, device)
                model, normalizer = model_info

                # Use appropriate dataloader logic for Equiformer
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

                dataloader = PyGDataLoader(id_dataset, batch_size=8, shuffle=False)

                predictions, targets, metadata = evaluate_equiformer_detailed(
                    model_info, dataloader, device, target_name, seed
                )

                df = pd.DataFrame(
                    {
                        "model": "equiformer",
                        "property": target_name,
                        "seed": seed,
                        "prediction": predictions,
                        "actual": targets,
                        "temperature": [m.get("temperature") for m in metadata],
                        "composition": [m.get("composition") for m in metadata],
                    }
                )

                df.to_csv(combined_path, mode="a", header=first_write, index=False)
                first_write = False
                print(
                    f"Saved results for regression/equiformer/{target_name}/{seed} to {combined_path}"
                )

            except Exception as e:
                print(
                    f"Error evaluating regression/equiformer/{target_name}/{seed}: {e}"
                )
                continue

    print(f"All Equiformer results saved to {combined_path}")


if __name__ == "__main__":
    main()
