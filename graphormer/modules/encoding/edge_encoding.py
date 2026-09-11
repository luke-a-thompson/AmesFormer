import torch
import torch.nn as nn

from graphormer.modules.model_data import ModelData


class EdgeEncoding(nn.Module):
    def __init__(self, edge_embedding_dim: int, max_path_distance: int):
        super().__init__()
        self.edge_embedding_dim = edge_embedding_dim
        self.max_path_distance = max_path_distance
        self.edge_vector = nn.Parameter(torch.zeros(max_path_distance, edge_embedding_dim))
        self.eps = 1e-9

    def forward(self, data: ModelData) -> ModelData:
        # Project each edge at each hop before gathering: avoid a B*N*N*L*D tensor.
        paths = data.edge_paths
        invalid = (paths < 0) | (paths >= data.edge_embedding.shape[1])
        scores = torch.einsum("bed,ld->bel", data.edge_embedding, self.edge_vector)
        batch = torch.arange(paths.shape[0], device=paths.device)[:, None, None, None]
        hop = torch.arange(paths.shape[-1], device=paths.device)
        scores = scores[batch, paths.masked_fill(invalid, 0), hop].masked_fill(invalid, 0)
        lengths = (~invalid).sum(dim=-1).float() + self.eps
        data.edge_encoding = scores.sum(dim=-1) / lengths
        return data
