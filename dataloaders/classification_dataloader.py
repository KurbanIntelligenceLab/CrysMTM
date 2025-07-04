import os
from typing import Any, Callable, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

# Add this import for PyG Data
try:
    from torch_geometric.data import Data as PyGData
except ImportError:
    PyGData = None


class ClassificationLoader(Dataset):
    """PyTorch Dataset for loading TDCM25 data with selectable modalities.
    Args:
        base_dir (str): Root directory of the dataset.
        temperature_filter (callable, optional): Function to filter temperatures.
        transform (callable, optional): Transformations to apply on images.
        modalities (list of str): Modalities to return. Any of 'image', 'xyz', 'text', 'element'.
        max_rotations (int, optional): Maximum number of rotations to include per temperature.
                                      If None, uses all available rotations.
        as_pyg_data (bool, optional): If True and modalities are ['xyz', 'element'], returns PyG Data object.
    """

    def __init__(
        self,
        base_dir: str,
        temperature_filter: Optional[Callable[[int], bool]] = None,
        transform: Optional[Callable] = None,
        modalities: Optional[List[str]] = None,
        max_rotations: Optional[int] = None,
        as_pyg_data: bool = False,
    ):
        self.base_dir = base_dir
        self.temperature_filter = temperature_filter
        self.transform = transform
        self.modalities = modalities or ["image", "xyz", "text", "element"]
        self.max_rotations = max_rotations
        self.as_pyg_data = as_pyg_data
        self.data = self._prepare_dataset()

    def _get_available_rotations(self, temp_dir: str) -> Dict[str, List[int]]:
        """Get available rotations for each modality in a temperature directory."""
        available_rotations = {}

        # Check images
        if "image" in self.modalities:
            images_dir = os.path.join(temp_dir, "images")
            if os.path.isdir(images_dir):
                image_files = [f for f in os.listdir(images_dir) if f.endswith(".png")]
                image_rotations = []
                for f in image_files:
                    if f.startswith("rot_") and f.endswith(".png"):
                        try:
                            rot_num = int(f[4:-4])  # Extract number from 'rot_X.png'
                            image_rotations.append(rot_num)
                        except ValueError:
                            continue
                available_rotations["image"] = sorted(image_rotations)

        # Check xyz files
        if "xyz" in self.modalities or "element" in self.modalities:
            xyz_dir = os.path.join(temp_dir, "xyz")
            if os.path.isdir(xyz_dir):
                xyz_files = [f for f in os.listdir(xyz_dir) if f.endswith(".xyz")]
                xyz_rotations = []
                for f in xyz_files:
                    if f.startswith("rot_") and f.endswith(".xyz"):
                        try:
                            rot_num = int(f[4:-4])  # Extract number from 'rot_X.xyz'
                            xyz_rotations.append(rot_num)
                        except ValueError:
                            continue
                available_rotations["xyz"] = sorted(xyz_rotations)

        # Check text files
        if "text" in self.modalities:
            text_dir = os.path.join(temp_dir, "text")
            if os.path.isdir(text_dir):
                text_files = [f for f in os.listdir(text_dir) if f.endswith(".txt")]
                text_rotations = []
                for f in text_files:
                    if f.startswith("rot_") and f.endswith(".txt"):
                        try:
                            rot_num = int(f[4:-4])  # Extract number from 'rot_X.txt'
                            text_rotations.append(rot_num)
                        except ValueError:
                            continue
                available_rotations["text"] = sorted(text_rotations)

        return available_rotations

    def _prepare_dataset(self) -> List[Dict[str, Any]]:
        data = []
        phases = ["anatase", "brookite", "rutile"]

        for phase in phases:
            for temp in range(0, 1001, 50):
                if self.temperature_filter and not self.temperature_filter(temp):
                    continue

                temp_dir = os.path.join(self.base_dir, phase, f"{temp}K")
                if not os.path.exists(temp_dir):
                    continue

                # Get available rotations for each modality
                available_rotations = self._get_available_rotations(temp_dir)

                # Find common rotations across all requested modalities
                if not available_rotations:
                    continue

                # Get the intersection of all available rotations
                common_rotations = set(available_rotations[self.modalities[0]])
                for modality in self.modalities[1:]:
                    if modality in available_rotations:
                        common_rotations = common_rotations.intersection(
                            set(available_rotations[modality])
                        )

                if not common_rotations:
                    continue

                # Sort and limit rotations if specified
                common_rotations = sorted(list(common_rotations))
                if self.max_rotations is not None:
                    common_rotations = common_rotations[: self.max_rotations]

                # Create entries for each common rotation
                for rotation in common_rotations:
                    entry = {
                        "phase": phase,
                        "temperature": temp,
                        "rotation": rotation,
                        "label": phase,
                    }

                    # Add file paths for each modality
                    if "image" in self.modalities:
                        entry["image_path"] = os.path.join(
                            temp_dir, "images", f"rot_{rotation}.png"
                        )
                    if "xyz" in self.modalities or "element" in self.modalities:
                        entry["xyz_path"] = os.path.join(
                            temp_dir, "xyz", f"rot_{rotation}.xyz"
                        )
                    if "text" in self.modalities:
                        entry["text_path"] = os.path.join(
                            temp_dir, "text", f"rot_{rotation}.txt"
                        )

                    data.append(entry)

        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Any:
        entry = self.data[idx]
        result = {}

        if self.as_pyg_data and set(self.modalities) == {"xyz", "element"}:
            if PyGData is None:
                raise ImportError("torch_geometric is required for as_pyg_data=True")
            # Build PyG Data object
            with open(entry["xyz_path"], "r") as f:
                xyz_lines = f.readlines()[2:]
            element_symbols = []
            xyz_coords = []
            for line in xyz_lines:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                elem, x, y, z = parts
                element_symbols.append(elem)
                xyz_coords.append([float(x), float(y), float(z)])
            # Map elements to atomic numbers (Ti=22, O=8)
            element_to_z = {"Ti": 22, "O": 8}
            z = torch.tensor(
                [element_to_z[el] for el in element_symbols], dtype=torch.long
            )
            pos = torch.tensor(xyz_coords, dtype=torch.float)
            label_map = {"anatase": 0, "brookite": 1, "rutile": 2}
            y = torch.tensor([label_map[entry["phase"]]], dtype=torch.long)
            return PyGData(z=z, pos=pos, y=y)

        if "image" in self.modalities:
            image = Image.open(entry["image_path"]).convert("RGB")
            if self.transform:
                image = self.transform(image)
            result["image"] = image

        if "xyz" in self.modalities or "element" in self.modalities:
            with open(entry["xyz_path"], "r") as f:
                xyz_lines = f.readlines()[2:]
            element_symbols = []
            xyz_coords = []
            for line in xyz_lines:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                elem, x, y, z = parts
                element_symbols.append(elem)
                xyz_coords.append([float(x), float(y), float(z)])
            if "xyz" in self.modalities:
                xyz_tensor = torch.tensor(xyz_coords, dtype=torch.float)
                result["xyz"] = xyz_tensor
            if "element" in self.modalities:
                result["element"] = element_symbols

        if "text" in self.modalities:
            with open(entry["text_path"], "r") as f:
                text_data = f.read()
            result["text"] = text_data

        # Always return label
        label_map = {"anatase": 0, "brookite": 1, "rutile": 2}
        result["label"] = label_map[entry["phase"]]

        return result
