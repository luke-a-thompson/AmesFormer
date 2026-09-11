"""Shortest paths over physical bonds, with a separate virtual node."""

import torch
from graphormer.shortest_paths import shortest_paths


def shortest_path_distance(edge_index: torch.Tensor, max_distance: int = 5, num_nodes: int | None = None):
    if not 2 <= max_distance <= 64:
        raise ValueError("max_distance must be between 2 and 64")
    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1 if edge_index.numel() else 0
    edges = [(0, 0)] + [(source + 1, target + 1) for source, target in edge_index.t().tolist()]
    # The backend infers node count from endpoints. A self-loop includes trailing isolated atoms;
    # it never occurs in shortest paths and needs no physical edge feature.
    nodes, paths = shortest_paths(edges + [(num_nodes, num_nodes)], max_distance)
    node_paths = torch.tensor(nodes, dtype=torch.long)
    edge_paths = torch.tensor(paths, dtype=torch.long)
    atoms = torch.arange(1, num_nodes + 1)
    node_paths[0, 1:, :2] = torch.stack((torch.zeros_like(atoms), atoms), dim=-1)
    node_paths[1:, 0, :2] = torch.stack((atoms, torch.zeros_like(atoms)), dim=-1)
    extra_edges = torch.arange(len(edges), len(edges) + num_nodes).unsqueeze(-1)
    edge_paths[0, 1:, :1] = extra_edges
    edge_paths[1:, 0, :1] = extra_edges
    return node_paths, edge_paths, extra_edges
