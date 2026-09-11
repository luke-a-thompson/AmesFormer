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
        if self.dataset_type is None:
            from graphormer.data.datasets.graphormer_dataset import FileDataset

            dataset = FileDataset(self.dataset_path, max_distance=self.max_path_distance)
            self.num_node_features = dataset.num_node_features
            self.num_edge_features = dataset.num_edge_features
            self.num_classes = dataset.num_classes
            if self.dataset_regime == DatasetRegime.TEST:
                return GraphormerDataLoader(dataset, batch_size=self.batch_size, **dataloader_optimization_params)
            valid_ids, train_ids = train_test_split(
                range(len(dataset)), test_size=self.test_size, random_state=self.random_state
            )
            return (
                GraphormerDataLoader(Subset(dataset, train_ids), batch_size=self.batch_size, shuffle=True, **dataloader_optimization_params),
                GraphormerDataLoader(Subset(dataset, valid_ids), batch_size=self.batch_size, **dataloader_optimization_params),
            )
        match self.dataset_type:
            case DatasetType.HANSEN:
                if self.test_size is None:
                    raise AttributeError("test_size is not defined for HansenDataset")
                if self.random_state is None:
                    raise AttributeError("random_state is not defined for HansenDataset")
                from graphormer.data.datasets.graphormer_dataset import FileDataset

                dataset = FileDataset(self.dataset_path, max_distance=self.max_path_distance)
                self.num_node_features = dataset.num_node_features
                self.num_edge_features = dataset.num_edge_features
                self.num_classes = dataset.num_classes

                if self.dataset_regime == DatasetRegime.TEST:
                    if len(dataset) <= 6513:
                        raise ValueError(
                            f"The {self.dataset_path.stem} benchmark test split starts at row 6513, "
                            f"but this dataset has only {len(dataset)} rows. Supply the full, ordered dataset."
                        )
                    test_loader = GraphormerDataLoader(
                        dataset[6513:],  # type: ignore
                        batch_size=self.batch_size,
                        **dataloader_optimization_params,
                    )
                    return test_loader

                test_ids, train_ids = train_test_split(
                    range(len(dataset[:6513])), test_size=self.test_size, random_state=self.random_state
                )
                train_loader = GraphormerDataLoader(
                    Subset(dataset, train_ids),  # type: ignore
                    batch_size=self.batch_size,
                    shuffle=True,
                    **dataloader_optimization_params,
                )
                test_loader = GraphormerDataLoader(
                    Subset(dataset, test_ids),  # type: ignore
                    batch_size=self.batch_size,
                    **dataloader_optimization_params,
                )

                return train_loader, test_loader

            case DatasetType.HONMA:
                from graphormer.data.datasets.graphormer_dataset import FileDataset

                dataset = FileDataset(self.dataset_path, max_distance=self.max_path_distance)
                self.num_node_features = dataset.num_node_features
                self.num_edge_features = dataset.num_edge_features
                self.num_classes = dataset.num_classes

                # len(Honma) = 13730
                if self.dataset_regime == DatasetRegime.TEST:
                    if len(dataset) <= 12140:
                        raise ValueError(
                            f"The {self.dataset_path.stem} benchmark test split starts at row 12140, "
                            f"but this dataset has only {len(dataset)} rows. Supply the full, ordered dataset."
                        )
                    test_loader = GraphormerDataLoader(
                        dataset[12140:],  # type: ignore
                        batch_size=self.batch_size,
                        **dataloader_optimization_params,
                    )
                    return test_loader

                test_ids, train_ids = train_test_split(
                    range(len(dataset[:12140])), test_size=self.test_size, random_state=self.random_state
                )
                train_loader = GraphormerDataLoader(
                    Subset(dataset, train_ids),  # type: ignore
                    batch_size=self.batch_size,
                    shuffle=True,
                    **dataloader_optimization_params,
                )
                test_loader = GraphormerDataLoader(
                    Subset(dataset, test_ids),  # type: ignore
                    batch_size=self.batch_size,
                    **dataloader_optimization_params,
                )

                return train_loader, test_loader

            case DatasetType.COMBINED:
                from graphormer.data.datasets.graphormer_dataset import FileDataset

                dataset = FileDataset(self.dataset_path, max_distance=self.max_path_distance)
                self.num_node_features = dataset.num_node_features
                self.num_edge_features = dataset.num_edge_features
                self.num_classes = dataset.num_classes

                # The publication test split requires the full, ordered dataset.
                if self.dataset_regime == DatasetRegime.TEST and len(dataset) <= 18652:
                    raise ValueError(
                        "The Combined benchmark test split starts at row 18652, but this dataset "
                        f"has only {len(dataset)} rows. Use `uv run inference --smiles CCO --name AmesFormer-Pro` for "
                        "pretrained predictions, or supply the full ordered benchmark dataset."
                    )

                # len(Combined) = 20242
                if self.dataset_regime == DatasetRegime.TEST:
                    test_loader = GraphormerDataLoader(
                        dataset[18652:],  # type: ignore
                        batch_size=self.batch_size,
                        **dataloader_optimization_params,
                    )
                    return test_loader

                test_ids, train_ids = train_test_split(
                    range(len(dataset[:18652])), test_size=self.test_size, random_state=self.random_state
                )
                train_loader = GraphormerDataLoader(
                    Subset(dataset, train_ids),  # type: ignore
                    batch_size=self.batch_size,
                    shuffle=True,
                    **dataloader_optimization_params,
                )
                test_loader = GraphormerDataLoader(
                    Subset(dataset, test_ids),  # type: ignore
                    batch_size=self.batch_size,
                    **dataloader_optimization_params,
                )

                return train_loader, test_loader

            case DatasetType.OGBG_MOLPCBA:
                from graphormer.data.datasets import OGBDataset

                dataset = OGBDataset(str(self.dataset_path.parent), max_distance=self.max_path_distance)
                self.num_node_features = dataset.num_node_features
                self.num_edge_features = dataset.num_edge_features

                split_idx = dataset.get_idx_split()

                train_loader = GraphormerDataLoader(
                    dataset[split_idx["train"]],  # type: ignore
                    batch_size=self.batch_size,
                    shuffle=True,
                    **dataloader_optimization_params,
                )
                valid_loader = GraphormerDataLoader(
                    dataset[split_idx["valid"]],  # type: ignore
                    batch_size=self.batch_size,
                    **dataloader_optimization_params,
                )

                return train_loader, valid_loader

            case DatasetType.TOX24:
                from graphormer.data.datasets.graphormer_dataset import FileDataset

                dataset = FileDataset(self.dataset_path, max_distance=self.max_path_distance)
                self.num_node_features = dataset.num_node_features
                self.num_edge_features = dataset.num_edge_features
                self.num_classes = dataset.num_classes

                # len(Tox24) = 1513, no labels 1014:
                if self.dataset_regime == DatasetRegime.TEST:
                    if len(dataset) <= 1014:
                        raise ValueError(
                            f"The {self.dataset_path.stem} benchmark test split starts at row 1014, "
                            f"but this dataset has only {len(dataset)} rows. Supply the full, ordered dataset."
                        )
                    test_loader = GraphormerDataLoader(
                        dataset[1014:],  # type: ignore
                        batch_size=self.batch_size,
                        **dataloader_optimization_params,
                    )
                    return test_loader

                test_ids, train_ids = train_test_split(
                    range(len(dataset[:1014])), test_size=self.test_size, random_state=self.random_state
                )
                train_loader = GraphormerDataLoader(
                    Subset(dataset, train_ids),  # type: ignore
                    batch_size=self.batch_size,
                    shuffle=True,
                    **dataloader_optimization_params,
                )
                test_loader = GraphormerDataLoader(
                    Subset(dataset, test_ids),  # type: ignore
                    batch_size=self.batch_size,
                    **dataloader_optimization_params,
                )

                return train_loader, test_loader

    def __str__(self) -> str:
        return f"{self.dataset_path} dataset"
