from pathlib import Path

from typing import Self, Tuple

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from graphormer.config.options import DatasetRegime, DatasetType
from graphormer.data.dataloader import GraphormerDataLoader


class DataConfig:
    def __init__(
        self,
        dataset_path: str,
        dataset_regime: DatasetRegime,
        batch_size: int,
        max_path_distance: int,
    ):
        self.dataset_path = Path(dataset_path).expanduser().resolve()
        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {self.dataset_path}")
        benchmark_names = {
            "combined": DatasetType.COMBINED,
            "honma_new": DatasetType.HONMA,
            "hansen_new": DatasetType.HANSEN,
            "tox24_all": DatasetType.TOX24,
        }
        if self.dataset_path.is_dir():
            if self.dataset_path.name != "ogbg_molpcba":
                raise ValueError("Pass a CSV/XLSX file, or the ogbg_molpcba dataset directory.")
            self.dataset_type = DatasetType.OGBG_MOLPCBA
        else:
            if self.dataset_path.suffix.lower() not in {".csv", ".xlsx"}:
                raise ValueError("Dataset must be a CSV or XLSX file.")
            self.dataset_type = benchmark_names.get(self.dataset_path.stem.lower())
        self.dataset_regime = dataset_regime
        self.batch_size = batch_size
        self.max_path_distance = max_path_distance
        self.test_size = None
        self.random_state = None
        self.num_node_features = None
        self.num_edge_features = None
        self.num_classes = None
        self.num_workers = 4
        self.prefetch_factor = 4

    def with_random_state(self, random_state: int) -> Self:
        self.random_state = random_state
        return self

    def with_test_size(self, test_size: float) -> Self:
        self.test_size = test_size
        return self

    def with_num_workers(self, num_workers: int) -> Self:
        self.num_workers = num_workers
        return self

    def with_prefetch_factor(self, prefect_factor: int) -> Self:
        self.prefetch_factor = prefect_factor
        return self

    def build(self) -> Tuple[GraphormerDataLoader, GraphormerDataLoader] | GraphormerDataLoader:
        """
        Builds the dataloaders for the dataset.

        Returns:
            IF TEST:
                A single test dataloader.
            ELSE:
                A tuple of train and validation dataloaders.
        """
        dataloader_optimization_params = {
            "pin_memory": True,
            "num_workers": self.num_workers,
            "prefetch_factor": self.prefetch_factor if self.num_workers > 0 else None,
            "persistent_workers": self.num_workers > 0,
        }
        if self.dataset_type == DatasetType.OGBG_MOLPCBA:
            from graphormer.data.datasets import OGBDataset

            dataset = OGBDataset(str(self.dataset_path.parent), max_distance=self.max_path_distance)
            split = dataset.get_idx_split()
            train_ids, valid_ids = split["train"], split["valid"]
        else:
            from graphormer.data.datasets.graphormer_dataset import FileDataset

            if self.dataset_type == DatasetType.HANSEN and self.dataset_regime != DatasetRegime.TEST:
                if self.test_size is None:
                    raise AttributeError("test_size is not defined for HansenDataset")
                if self.random_state is None:
                    raise AttributeError("random_state is not defined for HansenDataset")
            dataset = FileDataset(self.dataset_path, max_distance=self.max_path_distance)
            self.num_classes = dataset.num_classes
            cutoff = {
                DatasetType.HANSEN: 6513, DatasetType.HONMA: 12140,
                DatasetType.COMBINED: 18652, DatasetType.TOX24: 1014,
            }.get(self.dataset_type, len(dataset))
            if self.dataset_regime == DatasetRegime.TEST:
                train_ids = None
                valid_ids = range(len(dataset))
            else:
                # Preserve the existing convention: test_size is the training fraction.
                valid_ids, train_ids = train_test_split(
                    range(min(cutoff, len(dataset))), test_size=self.test_size, random_state=self.random_state
                )
        self.num_node_features = dataset.num_node_features
        self.num_edge_features = dataset.num_edge_features
        valid_loader = GraphormerDataLoader(
            Subset(dataset, valid_ids), batch_size=self.batch_size, **dataloader_optimization_params
        )
        if train_ids is None:
            return valid_loader
        return (
            GraphormerDataLoader(Subset(dataset, train_ids), batch_size=self.batch_size, shuffle=True, **dataloader_optimization_params),
            valid_loader,
        )

    def __str__(self) -> str:
        return f"{self.dataset_path} dataset"
