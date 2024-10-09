import torch
import torch.nn as nn

from graphormer.modules.model_data import ModelData


class EdgeEncoding(nn.Module):
    def __init__(self, edge_embedding_dim: int, max_path_distance: int):
        """
        Initializes a new instance of the EdgeEncoding.

        Args:
            edge_embedding_dim (int): The dimension of the edge embeddings.
            max_path_distance (int): The maximum path distance.

        """
        super().__init__()
        self.edge_embedding_dim = edge_embedding_dim
        self.max_path_distance = max_path_distance
        # Prior that further edges are less important than closer ones
        self.edge_vector = nn.Parameter(torch.zeros(self.max_path_distance, self.edge_embedding_dim))
        self.eps = 1e-9

    def forward(self, data: ModelData) -> ModelData:
        """
        :param data.edge_embedding: edge feature matrix, shape (batch_size, num_edges, edge_dim)
        :param data.edge_paths: pairwise node paths in edge indexes, shape (batch_size, num_nodes, num_nodes, max_path_length)
        :return: torch.Tensor, Edge Encoding
        """
        batch_size = data.edge_paths.shape[0]
        max_index = data.edge_embedding.shape[1] - 1  # Maximum valid index

        # Create a mask for invalid indices (less than 0 or greater than max_index)
        invalid_indices_mask = (data.edge_paths < 0) | (data.edge_paths > max_index)

        # Replace invalid indices with 0 to prevent indexing errors
        edge_paths_safe = data.edge_paths.clone()
        edge_paths_safe[invalid_indices_mask] = 0

        # Prepare batch indices for indexing
        batch_indices = torch.arange(batch_size, device=data.edge_paths.device).view(batch_size, 1, 1, 1).expand_as(data.edge_paths)

        # Get the edge embeddings for each edge in the paths
        assert data.edge_embedding is not None
        edge_path_embeddings = data.edge_embedding[batch_indices, edge_paths_safe, :]

        # Zero out embeddings corresponding to invalid indices
        edge_path_embeddings[invalid_indices_mask] = 0.0

        # Compute the path lengths (number of valid edges in the path)
        path_lengths = (~invalid_indices_mask).sum(dim=-1).float() + self.eps  # Convert to float for division

        # Compute the edge path encoding
        # edge_path_embeddings: (batch_size, num_nodes, num_nodes, max_path_length, edge_dim)
        # self.edge_vector: (max_path_length, edge_dim)
        edge_path_encoding = torch.einsum("bnmld,ld->bnm", edge_path_embeddings, self.edge_vector)

        # Compute the mean embedding based on path lengths
        data.edge_encoding = edge_path_encoding.div(path_lengths)

        return data
