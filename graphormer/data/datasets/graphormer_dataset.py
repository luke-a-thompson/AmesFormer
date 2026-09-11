import hashlib
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_smiles
from tqdm import tqdm
import os

from graphormer.data.data_cleaning import check_smiles_and_label, process


class GraphormerDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, max_distance: int = 5):
        self.max_distance = max_distance
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        raise NotImplementedError()

    @property
    def processed_file_names(self):
        raise NotImplementedError()

    def process(self):
        """
        Process the raw data and save the processed data.

        This method cleans the raw data, converts it into a format suitable for training,
        and saves the processed data to a .pt file.

        Returns:
            None
        """
        file_path = self.raw_paths[0]
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == ".xlsx":
            df = pd.read_excel(file_path)
        elif file_extension == ".csv":
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        missing = {"smiles", "ames"} - set(df.columns)
        if missing:
            raise ValueError(f"Dataset requires smiles and ames columns; missing: {', '.join(sorted(missing))}")

        if not df["ames"].isin([0, 1]).all():
            raise ValueError("The ames column must contain binary labels 0/1, without missing values")
        if not df["smiles"].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError("The smiles column must contain nonempty SMILES strings")

        data_list = []

        for smiles, ames in tqdm(
            zip(df["smiles"], df["ames"]), total=len(df), desc="Processing dataset", unit="SMILES"
        ):
            label = torch.tensor([ames], dtype=torch.float)

            warning = check_smiles_and_label(smiles, label)
            if warning:
                raise ValueError(warning.replace("WARN: ", "").replace(", skipped", ""))

            data = from_smiles(smiles)
            data.y = label
            data = process(data, self.max_distance)
            data_list.append(data)

        if not data_list:
            raise ValueError("Dataset contains no valid labeled molecules")
        torch.save(self.collate(data_list), self.processed_paths[0])



class FileDataset(GraphormerDataset):
    """Load a table directly, with a cache tied to its contents and preprocessing."""

    def __init__(self, path, max_distance: int = 5):
        self.path = Path(path).resolve()
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()[:16]
        self.cache_name = f"{self.path.name}-{digest}-v2-paths{max_distance}.pt"
        super().__init__(str(self.path.parent), max_distance=max_distance)

    @property
    def raw_dir(self):
        return str(self.path.parent)

    @property
    def raw_file_names(self):
        return [self.path.name]

    @property
    def processed_dir(self):
        return str(self.path.parent / ".amesformer-cache")

    @property
    def processed_file_names(self):
        return [self.cache_name]
