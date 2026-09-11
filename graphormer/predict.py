"""Run a pretrained AmesFormer checkpoint on SMILES strings."""

from pathlib import Path

import torch
from rdkit import Chem
from torch_geometric.utils import from_smiles

from graphormer.config.hparams import HyperparameterConfig
from graphormer.data.data_cleaning import process
from graphormer.data.dataloader import GraphormerDataLoader


@torch.inference_mode()
def predict_smiles(
    smiles: list[str],
    checkpoint: str | Path = "pretrained_models/AmesFormer-Pro.pt",
    device: str = "auto",
    batch_size: int = 32,
) -> list[dict[str, str | float]]:
    """Predict mutagenicity probabilities using a trusted model checkpoint."""
    if not smiles:
        raise ValueError("Provide at least one SMILES string")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for value in smiles:
        molecule = Chem.MolFromSmiles(value)
        if molecule is None or molecule.GetNumAtoms() == 0:
            raise ValueError(f"Invalid or empty SMILES: {value!r}")

    checkpoint = Path(checkpoint)
    hparams = HyperparameterConfig(
        name=checkpoint.stem,
        checkpoint_dir=str(checkpoint.parent),
        torch_device=device,
        temperature=1.0,
    )
    hparams.load_for_inference(verbose=False)
    torch.manual_seed(hparams.random_state)
    graphs = [process(from_smiles(value), hparams.max_path_distance) for value in smiles]
    model = (
        hparams.model_config()
        .with_node_feature_dim(graphs[0].num_node_features)
        .with_edge_feature_dim(graphs[0].num_edge_features)
        .with_output_dim(1)
        .build()
        .to(hparams.torch_device)
        .eval()
    )
    probabilities = []
    for batch in GraphormerDataLoader(graphs, batch_size=batch_size):
        logits = model(batch.to(hparams.torch_device))
        probabilities.extend(torch.sigmoid(logits / hparams.temperature).flatten().cpu().tolist())
    return [{"smiles": value, "probability": probability} for value, probability in zip(smiles, probabilities)]
