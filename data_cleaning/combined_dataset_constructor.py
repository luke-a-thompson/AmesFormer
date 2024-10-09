import pandas as pd
import rdkit.Chem as Chem
from rdkit import RDLogger
from tqdm import tqdm
import json

RDLogger.DisableLog("rdApp.*")  # type: ignore

hansen = pd.read_excel("datasets/Hansen_New.xlsx")
hansen["source"] = "hansen"

honma = pd.read_excel("/home/luke/ames_graphormer/data/raw/Honma_New.xlsx")
honma = honma[honma.columns[::-1]]  # Reverse columns
honma["source"] = "honma"

honma_train = honma[:-1589]
honma_test = honma[-1589:]

isssty = pd.read_excel("datasets/ISSSTY_fixed.xlsx")
isssty = isssty[["SMILES", "OVERALL"]]
isssty.columns = ["smiles", "ames"]
isssty["source"] = "isssty"
isssty['ames'] = isssty['ames'].replace({3: 1, 1: 0, 'Inc ': 0})  # Recoding
isssty = isssty.dropna()

eurl = pd.read_excel("datasets/eurl_ecvam.xls", skiprows=1)
eurl = eurl[["Smiles", "AMES Overall"]]
eurl.columns = ["smiles", "ames"]
eurl['ames'] = 0
eurl["source"] = "eurl"
eurl = eurl.dropna()


# Identify duplicate SMILES across all three datasets
hansen_smiles = set(hansen["smiles"])
honma_smiles = set(honma_train["smiles"])
isssty_smiles = set(isssty["smiles"])
eurl_smiles = set(eurl["smiles"])

combined = pd.concat([hansen, honma_train, isssty, eurl], ignore_index=True)

failed = 0

for index, row in tqdm(combined.iterrows(), total=len(combined)):
    try:
        molecule = Chem.MolFromSmiles(row["smiles"])
    except:
        print(row["smiles"])
        assert False, row["smiles"]
    if molecule is None:
        failed += 1
    else:  # Go to molecule and get the smiles - Canonicalises
        # combined.at[index, "molecule"] = molecule
        combined.at[index, "smiles"] = Chem.MolToSmiles(molecule)

print(f"Failed: {failed}")

def check_duplicates(dup_df: pd.DataFrame, print_disagreements: bool = False, save_disagreements: bool = False) -> pd.DataFrame:
    duplicates = dup_df[dup_df.duplicated(subset=["smiles"], keep=False)]
    disagreements = duplicates.groupby("smiles").filter(lambda x: x["ames"].nunique() > 1)

    if print_disagreements and not disagreements.empty:
        print("\nDuplicates with disagreements in 'ames' column:")
        for smiles, group in disagreements.groupby("smiles"):
            print(f"\nSMILES: {smiles}")
            print(group[["source", "ames"]])
    else:
        print("\nNo disagreements found in duplicate SMILES.")

    if save_disagreements:
        duplicates_json = {}
        for smiles, group in disagreements.groupby('smiles'):
            duplicates_json[smiles] = {
                source: ames
                for source, ames in zip(group['source'], group['ames'])
            }

        with open('duplicates_before_cleaning.json', 'w') as f:
            json.dump(duplicates_json, f, indent=2)

    return dup_df

check_duplicates(combined, save_disagreements=True)

def source_priority(source):
    priorities = {
        'honma': 0,
        'eurl': 1,
        'isssty': 2,
        'hansen': 3,
    }
    return priorities.get(source, len(priorities))  # Default to highest priority if source not in dict

# Drop duplicates, prioritizing 'isssty' source
combined_cleaned = combined.sort_values('source', key=lambda x: x.map(source_priority)) \
                           .drop_duplicates(subset=["smiles"], keep="first") \
                           .sort_index() \
                           .reset_index(drop=True) \
                           .dropna()
combined_cleaned["ames"] = combined_cleaned["ames"].astype(int)

check_duplicates(combined_cleaned, True, False)


# Late recoding
combined_cleaned['ames'] = combined_cleaned['ames'].replace({2.0: 0})

# Concatenating test set
combined_cleaned['split'] = 'Train/Validation'
honma_test['split'] = 'Test'

# Check if any of honma_test['smiles'] are in combined_cleaned
duplicate_smiles = set(honma_test['smiles']).intersection(set(combined_cleaned['smiles']))
if duplicate_smiles:
    print(f"Found {len(duplicate_smiles)} duplicate SMILES between test set and training set.")
    print("Removing these duplicates from the training set...")
    combined_cleaned = combined_cleaned[~combined_cleaned['smiles'].isin(duplicate_smiles)]
else:
    print("No duplicate SMILES found between test set and training set.")


combined_cleaned = pd.concat([combined_cleaned, honma_test])

combined_cleaned.to_csv("combined_datasets/Combined_2s_as_0s.csv", index=False)
combined_cleaned.to_excel("combined_datasets/Combined_2s_as_0s.xlsx", index=False)

# Calculate and print the lengths of train/validation and test sets
train_val_len = len(combined_cleaned[combined_cleaned['split'] == 'Train/Validation'])
test_len = len(combined_cleaned[combined_cleaned['split'] == 'Test'])

print(f"Length of Train/Validation set: {train_val_len}")
print(f"Length of Test set: {test_len}")

