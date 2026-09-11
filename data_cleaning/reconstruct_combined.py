"""Recover Combined from retained public rows and the original Honma workbook."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger, rdBase

TEST_ROWS = 1589
SOURCE_ORDER = ['hansen', 'honma', 'isssty', 'eurl']


def canonical_smiles(value):
    if not isinstance(value, str) or not value.strip():
        return None
    mol = Chem.MolFromSmiles(value)
    return Chem.MolToSmiles(mol) if mol is not None and mol.GetNumAtoms() else None


def annotate(frame, input_name):
    frame = frame.copy()
    frame['input'] = input_name
    frame['input_row'] = range(1, len(frame) + 1)
    frame['original_smiles'] = frame.smiles
    frame['canonical_smiles'] = frame.smiles.map(canonical_smiles)
    return frame


def reconstruct(public, honma, test_rows=TEST_ROWS):
    """Return an archival candidate and separate, validated training/test tables."""
    if not 0 < test_rows < len(honma):
        raise ValueError('Honma must contain both training and test rows')
    if not set(public.source).issubset(set(SOURCE_ORDER) - {'honma'}):
        raise ValueError('Public rows must come from Hansen, ISSSTY, or EURL')
    public = annotate(public, 'public')
    honma = annotate(honma.assign(source='honma'), 'honma')
    train_honma, test = honma.iloc[:-test_rows].copy(), honma.iloc[-test_rows:].copy()
    test['split'] = 'Test'
    if test.canonical_smiles.isna().any() or not test.ames.isin([0, 1]).all():
        raise ValueError('The historical test set contains invalid structures or labels; refusing to alter it')
    if test.canonical_smiles.duplicated().any():
        raise ValueError('The historical test set contains duplicate molecules')

    # Reinsert Honma between Hansen and ISSSTY, preserving source-relative row order.
    parts = [train_honma if source == 'honma' else public[public.source == source] for source in SOURCE_ORDER]
    pool = pd.concat(parts, ignore_index=True)
    pool['split'] = 'Train/Validation'
    pool['ames'] = pool.ames.replace({2: 0})
    if not pool.ames.dropna().isin([0, 1]).all():
        raise ValueError('Unexpected labels: expected 0/1, 2 (recoded to 0), or missing')

    # Reproduce the constructor's treatment of failed parses and raw test SMILES.
    historical = pool.copy()
    historical['smiles'] = historical.canonical_smiles.fillna(historical.original_smiles)
    priority = {'honma': 0, 'eurl': 1, 'isssty': 2, 'hansen': 3}
    historical = historical.sort_values('source', key=lambda s: s.map(priority))
    historical = historical.drop_duplicates('smiles').sort_index().dropna(subset=['smiles', 'ames'])
    historical = historical[~historical.smiles.isin(test.original_smiles)]
    historical = pd.concat([historical, test], ignore_index=True)
    historical['ames'] = historical.ames.astype(int)

    rejected = []
    def exclude(frame, mask, reason):
        rejected.append(frame.loc[mask].assign(reason=reason))
        return frame.loc[~mask].copy()

    pool = exclude(pool, pool.ames.isna(), 'missing_label')
    pool = exclude(pool, pool.canonical_smiles.isna(), 'invalid_smiles')
    pool = exclude(pool, pool.canonical_smiles.isin(test.canonical_smiles), 'test_overlap')
    pool = pool.sort_values('source', key=lambda s: s.map(priority), kind='stable')
    pool = exclude(pool, pool.canonical_smiles.duplicated(), 'duplicate_molecule').sort_index()
    for frame in [pool, test]:
        frame['smiles'] = frame.canonical_smiles
        frame['ames'] = frame.ames.astype(int)
    return historical, pool.reset_index(drop=True), test.reset_index(drop=True), pd.concat(rejected, ignore_index=True)


def fingerprint(path):
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--honma', type=Path, required=True)
    parser.add_argument('--test-reference', type=Path, required=True, help='Original Honma_Ames_Test.xlsx workbook')
    parser.add_argument('--public', type=Path, default=Path('data/raw/Combined.xlsx'))
    parser.add_argument('--output', type=Path, default=Path('data/reconstructed'))
    args = parser.parse_args()
    RDLogger.DisableLog('rdApp.*')
    public = pd.read_excel(args.public) if args.public.suffix.lower() == '.xlsx' else pd.read_csv(args.public)
    honma = pd.read_excel(args.honma)
    reference = pd.read_excel(args.test_reference)
    expected = honma.iloc[-TEST_ROWS:]
    reference_labels = reference['Ames result'].map({'A': 1, 'B': 1, 'C': 0})
    if not (expected.smiles.tolist() == reference.SMILES.tolist() and expected.ames.tolist() == reference_labels.tolist()):
        raise ValueError('Recovered Honma test rows do not match the independent test workbook in order and labels')
    historical, train, test, rejected = reconstruct(public, honma)
    combined = pd.concat([train, test], ignore_index=True)
    assert set(train.smiles).isdisjoint(test.smiles)
    assert not combined.smiles.duplicated().any()
    assert combined.ames.isin([0, 1]).all()
    columns = ['smiles', 'ames', 'source', 'split', 'original_smiles', 'input', 'input_row']
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = {'historical_candidate.csv': historical, 'combined_train_validation.csv': train,
               'honma_test.csv': test, 'combined_reconstructed.csv': combined}
    for name, frame in outputs.items():
        frame[columns].to_csv(args.output / name, index=False)
    rejected[columns + ['reason']].to_csv(args.output / 'excluded_rows.csv', index=False)
    historical_train = historical[historical.split != 'Test']
    summary = {
        'status': 'candidate reconstruction; original complete output unavailable for exact comparison',
        'rdkit_version': rdBase.rdkitVersion,
        'inputs': {name: fingerprint(path) for name, path in [('public', args.public), ('honma', args.honma), ('test_reference', args.test_reference)]},
        'rows': {name: len(frame) for name, frame in outputs.items()},
        'historical_training_rows': len(historical_train),
        'historical_invalid_smiles': int(historical.canonical_smiles.isna().sum()),
        'excluded': rejected.reason.value_counts().to_dict(),
        'training_sources': train.source.value_counts().to_dict(),
        'training_labels': train.ames.value_counts().to_dict(),
        'test_labels': test.ames.value_counts().to_dict(),
        'canonical_train_test_overlap': len(set(train.smiles) & set(test.smiles)),
        'test_reference_matches_order_and_labels': True,
        'output_sha256': {name: fingerprint(args.output / name)['sha256'] for name in [*outputs, 'excluded_rows.csv']},
    }
    (args.output / 'manifest.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({key: summary[key] for key in ['rows', 'historical_training_rows', 'excluded', 'canonical_train_test_overlap']}, indent=2))


if __name__ == '__main__':
    main()
