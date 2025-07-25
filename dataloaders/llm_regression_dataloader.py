import os
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from torch.utils.data import Dataset


class LLMLoader(Dataset):
    """PyTorch Dataset for LLM-style data.
    Returns a dict with text, image_path, xyz_path, phase, temperature, rotation, and regression labels.
    Args:
        label_dir (str): Directory containing *_labels.txt files and phase folders.
        temperature_filter (callable, optional): Function to filter temperatures.
        modalities (list of str): Modalities to return. Any of 'image', 'xyz', 'text'.
        max_rotations (int, optional): Maximum number of rotations to include per temperature.
    """

    def __init__(
        self,
        label_dir: str,
        temperature_filter: Optional[Callable[[int], bool]] = None,
        modalities: Optional[List[str]] = None,
        max_rotations: Optional[int] = None,
    ):
        self.label_dir = label_dir
        self.temperature_filter = temperature_filter
        self.modalities = modalities or ["image", "xyz", "text"]
        self.max_rotations = max_rotations
        self.label_data = self._load_label_data()
        self.data = self._prepare_dataset()

    def _load_label_data(self) -> Dict[str, Dict[int, Dict[str, float]]]:
        phases = ["anatase", "brookite", "rutile"]
        label_data = {phase: {} for phase in phases}

        # Load data from the CSV file we generated
        csv_file = os.path.join(self.label_dir, "labels.csv")
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")

        # Read the CSV file
        df = pd.read_csv(csv_file)

        # Process each row
        for _, row in df.iterrows():
            polymorph = row["Polymorph"].lower()  # Convert to lowercase to match phases
            temp_str = row["Temperature"]
            temp = (
                int(temp_str.replace("K", ""))
                if "K" in temp_str
                else int(float(temp_str))
            )
            parameter = row["Parameter"]
            value = float(row["Value"])

            # Initialize temperature dict if not exists
            if temp not in label_data[polymorph]:
                label_data[polymorph][temp] = {}

            # Map parameter names to match our expected format
            param_mapping = {
                "HOMO": "HOMO",
                "LUMO": "LUMO",
                "Eg": "Eg",
                "Ef": "Ef",
                "Et": "Et",
                "Eta": "Eta",
                "disp": "disp",
                "vol": "vol",
                "bond": "bond",
            }

            if parameter in param_mapping:
                label_data[polymorph][temp][param_mapping[parameter]] = value

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
        if "xyz" in self.modalities:
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
                        common_rotations = common_rotations.intersection(
                            set(available_rotations[modality])
                        )
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
                        entry["image_path"] = os.path.join(
                            temp_dir, "images", f"rot_{rotation}.png"
                        )
                    if "xyz" in self.modalities:
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

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.data[idx]
        result = {
            "phase": entry["phase"],
            "temperature": entry["temperature"],
            "rotation": entry["rotation"],
        }
        if "image_path" in entry:
            result["image_path"] = entry["image_path"]
        if "xyz_path" in entry:
            result["xyz_path"] = entry["xyz_path"]
        if "text_path" in entry:
            with open(entry["text_path"], "r") as f:
                result["text"] = f.read()
        # Add regression labels
        temp = entry["temperature"]
        phase = entry["phase"]
        label_dict = self.label_data[phase][temp]

        # Add all 9 targets from our CSV
        result["HOMO"] = label_dict["HOMO"]
        result["LUMO"] = label_dict["LUMO"]
        result["Eg"] = label_dict["Eg"]
        result["Ef"] = label_dict["Ef"]
        result["Et"] = label_dict["Et"]
        result["Eta"] = label_dict["Eta"]
        result["disp"] = label_dict["disp"]
        result["vol"] = label_dict["vol"]
        result["bond"] = label_dict["bond"]

        return result
