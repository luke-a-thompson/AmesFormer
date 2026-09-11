# AmesFormer

Graph transformer for mutagenicity prediction. [Paper](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/66fd0cd5cec5d6c142e23a70/original/amesformer-a-graph-transformer-neural-network-for-mutagenicity-prediction.pdf).

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run from the repository root:

```sh
uv sync                  # CPU
uv sync --extra cuda     # Optional: NVIDIA CUDA
```

Predict with the bundled AmesFormer-Pro model:

```sh
uv run --no-sync inference --smiles 'CCO' --smiles 'c1ccccc1' --name AmesFormer-Pro
```

Run inference on a dataset's test split:

```sh
uv run --no-sync inference --dataset data/raw/Combined.xlsx --name AmesFormer-Pro
```

- **Model:** inference requires `--name` or `--checkpoint`.
- **Dataset:** a CSV/XLSX file with `smiles` and `ames` (0/1) columns.
- **Splits:** files named `Combined`, `Honma_New`, `Hansen_New`, or `tox24_all` use benchmark splits. Other files use all rows for inference and a random train/validation split for training.
- **Combined:** benchmark inference needs the full, ordered dataset; the bundled workbook is too small.
- **OGB:** the legacy loader accepts an `ogbg_molpcba` directory, but its multitask labels are not supported by the binary AmesFormer training/inference pipeline.
- **Cache:** graphs are cached automatically in `.amesformer-cache/` beside the dataset.

Train on the bundled dataset:

```sh
uv run --no-sync train --dataset data/raw/Combined.xlsx --name my-model
```

`--no-sync` preserves your installed CUDA choice. Use a new `--name` for a fresh training run. See `inference --help` or `train --help` via `uv run --no-sync` for options; training defaults are in [default_hparams.toml](default_hparams.toml).

## Development

```sh
uv run --no-sync pytest
uv run --no-sync tensorboard --logdir runs
```

## Optional Rust acceleration

Python shortest paths are fast enough for typical molecules. For faster preprocessing, install [Rust](https://rustup.rs/) and a C compiler, then:

```sh
uv sync --extra rust                 # CPU
uv sync --extra cuda --extra rust    # CUDA
```

See [gnn-tools](gnn-tools/README.md) for details.
