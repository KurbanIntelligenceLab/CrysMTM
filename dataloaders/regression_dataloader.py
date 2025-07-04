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


class RegressionLoader(Dataset):
    """PyTorch Dataset for regression on TDCM25 data with selectable modalities and regression labels.
    Args:
        label_dir (str): Directory containing *_labels.txt files for each phase.
        temperature_filter (callable, optional): Function to filter temperatures.
        transform (callable, optional): Transformations to apply on images.
        modalities (list of str): Modalities to return. Any of 'image', 'xyz', 'text', 'element'.
        max_rotations (int, optional): Maximum number of rotations to include per temperature.
        as_pyg_data (bool, optional): If True and modalities are ['xyz', 'element'], returns PyG Data object.
    """

    def __init__(
        self,
        label_dir: str,
        temperature_filter: Optional[Callable[[int], bool]] = None,
        transform: Optional[Callable] = None,
        modalities: Optional[List[str]] = None,
        max_rotations: Optional[int] = None,
        as_pyg_data: bool = False,
    ):
        self.label_dir = label_dir
        self.temperature_filter = temperature_filter
        self.transform = transform
        self.modalities = modalities or ["image", "xyz", "text", "element"]
        self.max_rotations = max_rotations
        self.as_pyg_data = as_pyg_data
        self.label_data = self._load_label_data()
        self.data = self._prepare_dataset()

    def _load_label_data(self) -> Dict[str, Dict[int, Dict[str, float]]]:
        """Load regression labels from *_labels.txt files for each phase and temperature."""
        phases = ["anatase", "brookite", "rutile"]
        label_data = {phase: {} for phase in phases}
        label_names = ["HOMO", "LUMO", "Eg", "Ef", "Et"]
        for phase in phases:
            label_file = os.path.join(self.label_dir, f"{phase}_labels.txt")
            if not os.path.exists(label_file):
                continue
            with open(label_file, "r") as f:
                lines = f.readlines()
            for line in lines:
                if line.strip().startswith("#") or line.strip() == "" or line.strip().startswith("Temp"):
                    continue
                parts = line.split()
                if len(parts) < 6:
                    continue
                temp = int(parts[0].replace("K", "")) if "K" in parts[0] else int(float(parts[0]))
                label_data[phase][temp] = {name: float(parts[i+1]) for i, name in enumerate(label_names)}
        return label_data

    def _get_available_rotations(self, temp_dir: str) -> Dict[str, List[int]]:
        available_rotations = {}
        if "image" in self.modalities:
            images_dir = os.path.join(temp_dir, "images")
            if os.path.isdir(images_dir):
                image_files = [f for f in os.listdir(images_dir) if f.endswith(".png")]
                image_rotations = []
                for f in image_files:
                    if f.startswith("rot_") and f.endswith(".png"):
                        try:
                            rot_num = int(f[4:-4])
                            image_rotations.append(rot_num)
                        except ValueError:
                            continue
                available_rotations["image"] = sorted(image_rotations)
        if "xyz" in self.modalities or "element" in self.modalities:
            xyz_dir = os.path.join(temp_dir, "xyz")
            if os.path.isdir(xyz_dir):
                xyz_files = [f for f in os.listdir(xyz_dir) if f.endswith(".xyz")]
                xyz_rotations = []
                for f in xyz_files:
                    if f.startswith("rot_") and f.endswith(".xyz"):
                        try:
                            rot_num = int(f[4:-4])
                            xyz_rotations.append(rot_num)
                        except ValueError:
                            continue
                available_rotations["xyz"] = sorted(xyz_rotations)
        if "text" in self.modalities:
            text_dir = os.path.join(temp_dir, "text")
            if os.path.isdir(text_dir):
                text_files = [f for f in os.listdir(text_dir) if f.endswith(".txt")]
                text_rotations = []
                for f in text_files:
                    if f.startswith("rot_") and f.endswith(".txt"):
                        try:
                            rot_num = int(f[4:-4])
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
                temp_dir = os.path.join(self.label_dir, phase, f"{temp}K")
                if not os.path.exists(temp_dir):
                    continue
                available_rotations = self._get_available_rotations(temp_dir)
                if not available_rotations:
                    continue
                common_rotations = set(available_rotations[self.modalities[0]])
                for modality in self.modalities[1:]:
                    if modality in available_rotations:
                        common_rotations = common_rotations.intersection(set(available_rotations[modality]))
                if not common_rotations:
                    continue
                common_rotations = sorted(list(common_rotations))
                if self.max_rotations is not None:
                    common_rotations = common_rotations[: self.max_rotations]
                for rotation in common_rotations:
                    entry = {
                        "phase": phase,
                        "temperature": temp,
                        "rotation": rotation,
                    }
                    if "image" in self.modalities:
                        entry["image_path"] = os.path.join(temp_dir, "images", f"rot_{rotation}.png")
                    if "xyz" in self.modalities or "element" in self.modalities:
                        entry["xyz_path"] = os.path.join(temp_dir, "xyz", f"rot_{rotation}.xyz")
                    if "text" in self.modalities:
                        entry["text_path"] = os.path.join(temp_dir, "text", f"rot_{rotation}.txt")
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
            element_to_idx = {"Ti": 0, "O": 1}
            z = torch.tensor([element_to_idx[el] for el in element_symbols], dtype=torch.long)
            pos = torch.tensor(xyz_coords, dtype=torch.float)
            temp = entry["temperature"]
            phase = entry["phase"]
            label_dict = self.label_data[phase][temp]
            y = torch.tensor([
                label_dict["HOMO"],
                label_dict["LUMO"],
                label_dict["Eg"],
                label_dict["Ef"],
                label_dict["Et"],
            ], dtype=torch.float)
            data = PyGData(z=z, pos=pos)
            data.y = y.unsqueeze(0)  # shape [1, 5]
            return data
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
        temp = entry["temperature"]
        phase = entry["phase"]
        label_dict = self.label_data[phase][temp]
        result["regression_label"] = torch.tensor([
            label_dict["HOMO"],
            label_dict["LUMO"],
            label_dict["Eg"],
            label_dict["Ef"],
            label_dict["Et"],
        ], dtype=torch.float)
        return result 