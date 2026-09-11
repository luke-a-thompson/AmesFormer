import importlib.util
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('reconstruct_combined', Path(__file__).resolve().parents[1] / 'data_cleaning/reconstruct_combined.py')
reconstruction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reconstruction)


def test_reconstruct_preserves_test_and_prevents_canonical_leakage():
    public = pd.DataFrame({
        'smiles': ['CCO', 'CC', 'invalid', 'C#N'],
        'ames': [0, 0, 0, 1], 'source': ['hansen', 'hansen', 'isssty', 'eurl'],
    })
    honma = pd.DataFrame({'smiles': ['CC', 'N', 'O', 'OCC', 'CCC'], 'ames': [1, 2, None, 0, 1]})
    historical, train, test, rejected = reconstruction.reconstruct(public, honma, test_rows=2)
    # The original script compares canonical training SMILES to raw test SMILES,
    # leaving ethanol in both partitions when the test spells it OCC.
    assert 'CCO' in historical[historical.split != 'Test'].smiles.tolist()
    assert historical.iloc[-2:].smiles.tolist() == ['OCC', 'CCC']
    assert test.smiles.tolist() == ['CCO', 'CCC']
    assert test.original_smiles.tolist() == ['OCC', 'CCC']
    assert test.ames.tolist() == [0, 1]
    assert train.smiles.tolist() == ['CC', 'N', 'C#N']
    assert train.ames.tolist() == [1, 0, 1]  # Honma wins conflicting public label; 2 recodes to 0.
    assert set(train.smiles).isdisjoint(test.smiles)
    assert rejected.reason.value_counts().to_dict() == {'missing_label': 1, 'invalid_smiles': 1, 'test_overlap': 1, 'duplicate_molecule': 1}
    assert rejected[rejected.reason == 'test_overlap'].input_row.tolist() == [1]


@pytest.mark.parametrize('smiles,labels', [(['invalid'], [0]), (['CCO'], [None]), (['CCO', 'OCC'], [0, 0])])
def test_reconstruction_refuses_to_change_invalid_test_set(smiles, labels):
    public = pd.DataFrame({'smiles': ['C#N'], 'ames': [0], 'source': ['hansen']})
    honma = pd.DataFrame({'smiles': ['CC', *smiles], 'ames': [1, *labels]})
    with pytest.raises(ValueError, match='historical test set'):
        reconstruction.reconstruct(public, honma, test_rows=len(smiles))
