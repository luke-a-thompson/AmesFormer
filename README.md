# AmesFormer
AmesFormer: [A graph transformer neural network for state-of-the-art mutagenicity prediction](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/66fd0cd5cec5d6c142e23a70/original/amesformer-a-graph-transformer-neural-network-for-mutagenicity-prediction.pdf).

## Key Contributions
* We achieve state-of-the-art mutagenicity prediction on a standardised Ames dataset.
* We provide a large, clean open-source dataset of Ames mutagenicity.

## Understanding this Repository
* `data/` is empty, users should place their own dataset here, or use our provided `combined.csv` dataset.
* `data_cleaning/` contains the code used to create `combined.csv` from the individual raw datasets and code to generate some of our results figures.
    * Our Ames dataset, excluding the proprietary "Honma" components, is available in `datacleaning/Combined_2s_as_0s_publication.csv`
* `gnn-tools/` is our custom Rust library for calculating the shortest path distance (SPD) for the edge and spatial encoding modules.
* `graphormer/` contains the AmesFormer model, in essence a full reimplimentation of Graphormer in PyTorch Geometric.
* `hparams/` contains the hyperparameter configurations we used in our experiments. **The one used for our final model for which results are reported is `hparams/best_32_1_5e4.toml`**.
* `pretrained_models/` contains the saved model checkpoints for our final AmesFormer model.
* `tests/` contains unit tests for attention, encodings, etc.

# Installation
## Requirements
This repository includes some tools which are built using [Rust](https://www.rust-lang.org/) and create python bindings with [Maturin](https://github.com/PyO3/maturin).  These must both be installed in order to build from source.

## With Poetry
Installation is simplest with [Poetry](https://python-poetry.org/docs/). Run:
- `poetry lock --no-update` to gather the required information and populate caches
- `poetry install` to furnish a virtual environment.
- `poetry run inference --dataset Combined --name AmesFormer-Pro` to run inference our best model.
- `poetry run train --dataset Combined` to begin training the model.  See `poetry run train --help` for options.

# Visualization
## Requirements
Visualization is provided with `tensorboard`.  To see local readouts during training, first install `tensorboard` on your system.  We recommend via [pipx](https://github.com/pypa/pipx):
- `pipx install tensorboard`

## Local Server
You can start a local tensorboard server via the following command:
- `tensorboard --logdir=<logdir>`, where `<logdir>` is the path to the tensorboard logs.  By default, these are created in the `runs` folder.


## Layers of Graphormer Implemented
1. Centrality Encoding
2. Spatial Encoding
3. Edge Encoding
4. Multi-Head Self-Attention
5. VNODE global attention