import os
import pickle

import pandas as pd
import torch
from torch.utils.data import DataLoader

from configs.regression_config import (
    BASE_DIR,
    BATCH_SIZE,
    ID_TEMPS,
    MAX_ROTATIONS,
    NORMALIZATION_METHOD,
    OOD_TEMPS,
    SEEDS,
    TARGET_PROPERTIES,
)
from dataloaders.regression_dataloader import RegressionLoader
from models.regression.clip_regressor import DEFAULT_MODEL_NAME, create_clip_regressor

TEST_TEMPS = ID_TEMPS + OOD_TEMPS


def load_clip_model(target_name, seed, device):
    """Load trained CLIP model."""
    model_path = os.path.join(
        "results", f"regression/clip/{target_name}", str(seed), "best_model.pth"
    )

    # Load normalizer if available
    normalizer_path = os.path.join(
        "results", f"regression/clip/{target_name}", str(seed), "normalizer.pkl"
    )
    normalizer = None
    if os.path.exists(normalizer_path):
        with open(normalizer_path, "rb") as f:
            normalizer = pickle.load(f)

    model = create_clip_regressor(
        model_name=DEFAULT_MODEL_NAME, freeze_backbone=True
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model, normalizer


def collate_fn_clip(batch, processor):
    """Collate function for CLIP evaluation."""
    images = [item["image"] for item in batch]
    texts = [item["text"] for item in batch]
    regression_labels = torch.stack([item["regression_label"] for item in batch], dim=0)

    # Preserve metadata from the original items
    temperatures = [item.get("temperature", None) for item in batch]
    phases = [item.get("phase", None) for item in batch]

    # Process text and images separately to avoid processor compatibility issues
    try:
        # Try the standard approach first
        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
    except (AttributeError, TypeError):
        # Fallback: process text and images separately
        text_inputs = processor.tokenizer(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        image_inputs = processor.image_processor(images, return_tensors="pt")
        inputs = {**text_inputs, **image_inputs}

    result = {
        **inputs,
        "regression_label": regression_labels,
        "temperatures": temperatures,
        "phases": phases,
    }
    return result


def evaluate_clip_detailed(model_info, dataloader, device, target_name, seed):
    """Evaluate CLIP model and return detailed predictions with metadata."""
    model, normalizer = model_info
    model.eval()

    predictions = []
    targets = []
    metadata = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # CLIP specific logic - uses multimodal input (image + text)
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)

            outputs = model(batch)

            # Get target for the specific property
            target_index = TARGET_PROPERTIES.index(target_name)
            labels = batch["regression_label"]
            if labels.dim() == 2:
                target = labels[:, target_index].unsqueeze(1)
            else:
                target = labels.unsqueeze(1)

            # Keep both predictions and targets in normalized scale for fair comparison
            # The model outputs are already in normalized scale
            # The targets from the dataloader are now also in normalized scale
            pred_batch = outputs.cpu().numpy()
            target_batch = target.cpu().numpy()

            # Extract metadata - one entry per sample in the batch
            batch_size = len(batch["temperatures"])
            # Only use the first batch_size predictions/targets
            predictions.extend(pred_batch.flatten()[:batch_size])
            targets.extend(target_batch.flatten()[:batch_size])

            for i in range(batch_size):
                temperature = batch["temperatures"][i]
                composition = batch["phases"][i]
                if temperature is None or composition is None:
                    print(f"WARNING: Missing metadata at batch {batch_idx}, sample {i}")
                metadata.append(
                    {
                        "model": "clip",
                        "target": target_name,
                        "seed": seed,
                        "batch_idx": i,
                        "sample_idx": batch_idx * BATCH_SIZE + i,
                        "temperature": temperature,
                        "composition": composition,
                    }
                )

    return predictions, targets, metadata


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    output_dir = "results/detailed_analysis"
    os.makedirs(output_dir, exist_ok=True)
    combined_path = os.path.join(output_dir, "clip_predictions.csv")
    first_write = not os.path.exists(combined_path)

    for target_name in TARGET_PROPERTIES:
        for seed in SEEDS:
            try:
                model_path = os.path.join(
                    "results",
                    f"regression/clip/{target_name}",
                    str(seed),
                    "best_model.pth",
                )
                if not os.path.exists(model_path):
                    print(f"Model not found: {model_path}")
                    continue

                model_info = load_clip_model(target_name, seed, device)
                model, normalizer = model_info

                # Get processor from model
                processor = model.get_processor()

                # Use appropriate dataloader logic for CLIP (multimodal)
                id_dataset = RegressionLoader(
                    label_dir=BASE_DIR,
                    temperature_filter=lambda temp: temp in TEST_TEMPS,
                    modalities=["image", "text"],
                    transform=None,
                    max_rotations=MAX_ROTATIONS,
                    normalize_labels=True,  # Enable normalization
                    normalization_method=NORMALIZATION_METHOD,
                    fit_normalizer_on_data=False,
                )

                # Set the normalizer on the dataset if available
                if normalizer is not None:
                    id_dataset.set_normalizer(normalizer)

                # Use the same collate function as training
                dataloader = DataLoader(
                    id_dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=lambda b: collate_fn_clip(b, processor),
                )

                predictions, targets, metadata = evaluate_clip_detailed(
                    model_info, dataloader, device, target_name, seed
                )

                df = pd.DataFrame(
                    {
                        "model": "clip",
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
                    f"Saved results for regression/clip/{target_name}/{seed} to {combined_path}"
                )

            except Exception as e:
                print(f"Error evaluating regression/clip/{target_name}/{seed}: {e}")
                continue

    print(f"All CLIP results saved to {combined_path}")


if __name__ == "__main__":
    main()
