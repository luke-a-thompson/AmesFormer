from typing import Optional

import torch
from rdkit import Chem
from torch_geometric.utils import degree

from graphormer.functional import shortest_path_distance


def check_smiles_and_label(smiles, label) -> Optional[str]:
    if torch.isnan(label):
        return f"WARN: No label for {smiles}, skipped"

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return f"WARN: Invalid SMILES {smiles}, skipped"

    return None


def process(data, max_distance: int):
    num_atoms = data.num_nodes
    node_paths, edge_paths, _ = shortest_path_distance(data.edge_index, max_distance, num_atoms)
    # Centrality describes physical bonds; the virtual node has no physical degree.
    atom_degrees = torch.stack([degree(data.edge_index[0], num_atoms), degree(data.edge_index[1], num_atoms)], dim=-1)
    data.degrees = torch.cat((torch.zeros(1, 2), atom_degrees))
    virtual_edges = torch.stack((torch.zeros(num_atoms, dtype=torch.long), torch.arange(1, num_atoms + 1)))
    data.edge_index = torch.cat((torch.zeros(2, 1, dtype=torch.long), data.edge_index + 1, virtual_edges), dim=1)
    data.x = torch.cat((torch.full((1, data.x.shape[1]), -1), data.x))
    data.edge_attr = torch.cat((
        torch.full((1, data.edge_attr.shape[1]), -1), data.edge_attr,
        torch.full((num_atoms, data.edge_attr.shape[1]), -1),
    ))
    data.node_paths = node_paths.flatten(0, 1)
    data.edge_paths = edge_paths.flatten(0, 1)
    return data
