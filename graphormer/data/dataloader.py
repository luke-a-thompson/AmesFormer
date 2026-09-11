from typing import Any, List, Optional, Sequence, Union, override

import torch
import torch.nn.utils.rnn as rnn
from torch_geometric.data import Data
from torch_geometric.loader.dataloader import BaseData, Collater, DataLoader, Dataset, DatasetAdapter


class GraphormerDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        kwargs.pop("collate_fn", None)

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            **kwargs,
        )
        self.collate_fn = GraphormerCollater(dataset, follow_batch, exclude_keys)


class GraphormerBatch(Data):
    def __init__(self, *args, **kwargs):
        self.node_paths: Optional[torch.Tensor] = None
        self.edge_paths: Optional[torch.Tensor] = None
        self.degrees: Optional[torch.Tensor] = None
        super().__init__(*args, **kwargs)


class GraphormerCollater(Collater):
    @override
    def __call__(self, batch: List[Any]) -> GraphormerBatch:
        data: GraphormerBatch = super().__call__(batch)
        assert data.x is not None
        assert data.edge_index is not None
        assert data.edge_attr is not None
        assert data.node_paths is not None
        assert data.edge_paths is not None
        assert data.degrees is not None
        x = data.x.float()
        node_paths = data.node_paths

        degrees = data.degrees
        ptr = data.ptr

        node_subgraphs = []
        edge_attr_subgraphs = []
        node_paths_subgraphs = []
        edge_paths_subgraphs = []
        degree_subgraphs = []
        subgraph_idxs = torch.stack([ptr[:-1], ptr[1:]], dim=1)
        max_nodes = 0
        path_length = node_paths.shape[-1]
        for graph, idx_range in zip(batch, subgraph_idxs.tolist()):
            subgraph = x[idx_range[0] : idx_range[1]]
            num_nodes = idx_range[1] - idx_range[0]
            if num_nodes > max_nodes:
                max_nodes = num_nodes

            node_subgraphs.append(subgraph)
            degree_subgraphs.append(degrees[idx_range[0] : idx_range[1], :])

            # Virtual edge attributes have their own boundaries, independent of edge_index.
            edge_attr_subgraphs.append(graph.edge_attr.float())
            # Legacy preprocessing stores auxiliary paths beyond num_nodes**2.
            # Apply the single-graph slicing convention separately to each graph;
            # cumulative num_nodes**2 offsets mix paths from different molecules.
            num_paths = num_nodes**2
            node_paths_subgraphs.append(graph.node_paths[:num_paths].reshape(num_nodes, num_nodes, path_length))
            edge_paths_subgraphs.append(graph.edge_paths[:num_paths].reshape(num_nodes, num_nodes, path_length))
        target_shape = (max_nodes, max_nodes, path_length)

        node_subgraphs_padded = []
        edge_subgraphs_padded = []
        for node_subgraph, edge_subgraph in zip(node_paths_subgraphs, edge_paths_subgraphs):
            pad_bottom = target_shape[0] - node_subgraph.shape[0]
            pad_right = target_shape[1] - node_subgraph.shape[1]

            # (left, right, top, bottom, front, back)
            pad = (0, pad_right, 0, pad_bottom)
            node_subgraphs_padded.append(
                torch.nn.functional.pad(node_subgraph.permute(2, 0, 1), pad, mode="constant", value=-1).permute(1, 2, 0)
            )
            edge_subgraphs_padded.append(
                torch.nn.functional.pad(edge_subgraph.permute(2, 0, 1), pad, mode="constant", value=-1).permute(1, 2, 0)
            )

        node_paths = torch.stack(node_subgraphs_padded).long()
        edge_paths = torch.stack(edge_subgraphs_padded).long()

        data.x = rnn.pad_sequence(node_subgraphs, batch_first=True, padding_value=-2)
        data.degrees = rnn.pad_sequence(degree_subgraphs, batch_first=True, padding_value=-1).transpose(1, 2).long()
        data.edge_attr = rnn.pad_sequence(edge_attr_subgraphs, batch_first=True, padding_value=-1)
        data.node_paths = node_paths
        data.edge_paths = edge_paths
        return data
