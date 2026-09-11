# AmesFormer

Graph transformer for mutagenicity prediction. [Paper](https://pubs.acs.org/crtoec/article/38/7/1167/3764632/AmesFormer-State-of-the-Art-Mutagenicity).

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run from the repository root:

```sh
uv sync                  # CPU
uv sync --extra cuda     # Optional: NVIDIA CUDA
```

**Model update:** AmesFormer-Pro has been retrained with the graph and attention fixes: held-out ROC-AUC 0.786 and balanced accuracy 72.8%. See [retraining results](reports/retraining-results.md). AmesFormer-Honma still contains legacy weights and needs revalidation.

Predict with the bundled AmesFormer-Pro model:

```sh
uv run --no-sync inference --smiles 'CCO' --smiles 'c1ccccc1' --name AmesFormer-Pro
```

Run inference on every row of a dataset:

```sh
uv run --no-sync inference --dataset data/raw/Combined.xlsx --name AmesFormer-Pro
```

- **Model:** inference requires `--name` or `--checkpoint`.
- **Dataset:** a CSV/XLSX file with `smiles` and `ames` (0/1) columns.
- **Splits:** inference uses all rows. Training uses benchmark training portions for `Combined`, `Honma_New`, `Hansen_New`, and `tox24_all`; other filenames use a random train/validation split. Pass only held-out rows when evaluating performance.
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

## Citation

If you use AmesFormer, please cite the published paper:

```bibtex
@article{thompson2025amesformer,
  author = {Thompson, Luke A. and Evans, Josiah G. and Matthews, Slade T.},
  title = {{AmesFormer}: State-of-the-Art Mutagenicity Prediction with Graph Transformers},
  journal = {Chemical Research in Toxicology},
  year = {2025},
  volume = {38},
  number = {7},
  pages = {1167--1182},
  doi = {10.1021/acs.chemrestox.4c00466},
  url = {https://doi.org/10.1021/acs.chemrestox.4c00466}
}
```
