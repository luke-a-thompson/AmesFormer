# gnn-tools

Optional Rust accelerator for AmesFormer's shortest-path preprocessing. The default AmesFormer install uses the compatible Python implementation in `graphormer/shortest_paths.py` and does not build this package.

To enable it, install [Rust](https://rustup.rs/) and a C linker/compiler (for example, `build-essential` on Debian/Ubuntu, Xcode Command Line Tools on macOS, or Visual Studio C++ Build Tools on Windows), then run from the AmesFormer repository root:

```sh
uv sync --extra rust
uv run --extra rust pytest
```

Python 3.12 is required by this package's PyO3 version. uv installs Maturin automatically in the isolated build environment; no separate Maturin installation or Poetry environment is needed. Cargo dependencies are pinned in `Cargo.lock`.

The `shortest_paths(edges, max_path_len)` function returns padded node and edge paths for an undirected graph. Nodes are inferred from the highest endpoint index, and path lengths range from 0 to 64. Neighbor and edge selection order follows Petgraph, including when both orientations of a molecular bond are present. The integration tests compare these outputs with the Python fallback.
