import pandas as pd
from tqdm import tqdm
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")  # type: ignore

def canonicalize_smiles(df):
    failed = 0
    for index, row in tqdm(df.iterrows(), total=len(df)):
        try:
            molecule = Chem.MolFromSmiles(row["smiles"])
        except:
            print(row["smiles"])
            assert False, row["smiles"]
        if molecule is None:
            failed += 1
        else:  # Go to molecule and get the smiles - Canonicalises
            df.at[index, "smiles"] = Chem.MolToSmiles(molecule)

    print(f"Failed: {failed}")
    return df

honma = pd.read_excel("/home/luke/ames_graphormer/data/raw/Honma_New.xlsx")
# honma = canonicalize_smiles(honma)
honma_train = honma[:-1589]
honma_test = honma[-1589:]

combined = pd.read_excel("/home/luke/ames_graphormer/data/raw/Combined_2s_as_0s.xlsx")
# combined = canonicalize_smiles(combined)
combined_train = combined[combined['split'] == 'Train/Validation']
combined_test = combined[combined['split'] == 'Test']
assert len(combined_test) == 1589

# assert len(combined_test) == len(honma_test), f"Length of combined test: {len(combined_test)} != Length of Honma test {len(honma_test)}"

combined_test_smiles = set(combined_test['smiles'])
honma_test_smiles = set(honma_test['smiles'])

# SMILES in combined_test but not in honma_test
combined_only = combined_test_smiles - honma_test_smiles
assert len(combined_only) == 0, f"Expected 0 SMILES in combined_test but not in honma_test, but found {len(combined_only)}"
print(f"Number of SMILES in combined_test but not in honma_test: {len(combined_only)}")

# SMILES in honma_test but not in combined_test
honma_only = honma_test_smiles - combined_test_smiles
print(f"Number of SMILES in honma_test but not in combined_test: {len(honma_only)}")

# Check for duplicates
duplicates = combined_test_smiles.intersection(honma_test_smiles)
print(f"Number of duplicate SMILES between combined_test and honma_test: {len(duplicates)}")
assert len(duplicates) == len(honma_test), f"Expected {len(honma_test)} duplicates, but found {len(duplicates)}"