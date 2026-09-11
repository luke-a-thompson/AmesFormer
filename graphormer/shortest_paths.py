"""Shortest paths with the original Rust backend's ordering and padding."""

from collections import deque


def python_shortest_paths(
    edges: list[tuple[int, int]], max_path_len: int
) -> tuple[list[list[list[int]]], list[list[list[int]]]]:
    """Return padded node/edge paths, retaining Petgraph's tie-breaking order.

    Molecular bonds occur in both directions. Petgraph visits outgoing edges
    before incoming edges, in reverse insertion order within each group, and
    selects the first matching edge in that order. Keeping this convention is
    necessary for compatibility with the pretrained model's edge encodings.
    """
    if not 0 <= max_path_len <= 64:
        raise ValueError("max_path_len must be between 0 and 64")
    if any(source < 0 or target < 0 for source, target in edges):
        raise ValueError("node indices must be nonnegative")
    num_nodes = 1 + max((max(edge) for edge in edges), default=-1)
    outgoing = [[] for _ in range(num_nodes)]
    incoming = [[] for _ in range(num_nodes)]
    for edge_id, (source, target) in enumerate(edges):
        outgoing[source].append((target, edge_id))
        incoming[target].append((source, edge_id))
    neighbors = [list(reversed(out)) + list(reversed(inc)) for out, inc in zip(outgoing, incoming)]

    node_paths = []
    edge_paths = []
    for source in range(num_nodes):
        nodes = [[] for _ in range(num_nodes)]
        paths = [[] for _ in range(num_nodes)]
        nodes[source] = [source]
        queue = deque([source])
        while queue:
            current = queue.popleft()
            for neighbor, edge_id in neighbors[current]:
                if nodes[neighbor]:
                    continue
                # Rust stores at most 64 nodes (63 edges), then slices the paths.
                nodes[neighbor] = (nodes[current] + [neighbor])[:64]
                paths[neighbor] = (paths[current] + [edge_id])[:63]
                queue.append(neighbor)
        node_paths.append([(path + [-1] * max_path_len)[:max_path_len] for path in nodes])
        edge_paths.append([(path + [-1] * max_path_len)[:max_path_len] for path in paths])
    return node_paths, edge_paths


try:
    from gnn_tools import shortest_paths
except ModuleNotFoundError as error:
    if error.name != "gnn_tools":
        raise
    shortest_paths = python_shortest_paths
