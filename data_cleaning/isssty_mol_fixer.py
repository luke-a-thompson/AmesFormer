from rdkit import Chem
import pandas as pd
from tqdm import tqdm

# Load the SDF file
supplier = Chem.SDMolSupplier('datasets/ISSSTY.sdf')
isssty = pd.read_excel("datasets/ISSSTY_broken.xls")
assert len(supplier) == len(isssty), "The number of molecules in the SDF file and the Excel file do not match."

for index, mol in tqdm(enumerate(supplier)):
    if mol is not None:
        smiles = Chem.MolToSmiles(mol)
        isssty.at[index, "SMILES"] = smiles
    else:
        print(f"Molecule {index} of {len(isssty)} is None.")
        isssty.at[index, "SMILES"] = None
        # raise ValueError(f"Molecule {index} of {len(isssty)} is None, SMILES replacement out of sync.")

print(isssty["SMILES"].isna().sum())

isssty.to_excel("datasets/ISSSTY_fixed.xlsx", index=False)


# The .sdf file is in the same order as the excel file with invalid smiles
# I verified these are the same by converting the first broken smiles to a correct smiles using https://apps.ideaconsult.net/data/ui/toxtree
# I then compared this to the first smiles from the .sdf file
# They are the same, so the .sdf file is in the same order as the excel file with invalid smiles