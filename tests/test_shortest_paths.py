import random

import pytest

from graphormer.shortest_paths import python_shortest_paths


def test_disconnected_nodes_and_padding():
    nodes, edges = python_shortest_paths([(0, 1), (2, 2)], 3)
    assert nodes[0] == [[0, -1, -1], [0, 1, -1], [-1, -1, -1]]
    assert edges[0] == [[-1, -1, -1], [0, -1, -1], [-1, -1, -1]]
    assert nodes[2][2] == [2, -1, -1]
    assert python_shortest_paths([], 3) == ([], [])


def test_truncation_preserves_extra_edge():
    nodes, edges = python_shortest_paths([(0, 1), (1, 2), (2, 3)], 2)
    assert nodes[0][3] == [0, 1]
    assert edges[0][3] == [0, 1]


@pytest.mark.parametrize("length", [-1, 65])
def test_invalid_path_length(length):
    with pytest.raises(ValueError):
        python_shortest_paths([(0, 1)], length)


@pytest.mark.parametrize("length", [0, 1, 5, 64])
def test_rust_parity(length):
    native = pytest.importorskip("gnn_tools")
    rng = random.Random(42)
    for _ in range(30):
        edges = [(rng.randrange(12), rng.randrange(12)) for _ in range(30)]
        assert python_shortest_paths(edges, length) == native.shortest_paths(edges, length)
    chain = [(i, i + 1) for i in range(70)]
    assert python_shortest_paths(chain, length) == native.shortest_paths(chain, length)
