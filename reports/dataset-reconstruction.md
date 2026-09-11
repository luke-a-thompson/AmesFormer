# Combined dataset reconstruction

The recovered Honma workbook and the repository's retained public rows produce a historical candidate with 22,519 training/validation rows and the original 1,589 test rows. Replaying the original constructor on these retained inputs matches the candidate row for row. The complete original output is unavailable, so exact historical equivalence remains unverified.

After the legacy preprocessor skips two invalid structures, the candidate has 22,517 training/validation graphs. Pro's saved settings (seed 42, training fraction 0.999, batch size 32) produce 22,495 training examples and 703 batches per epoch, matching the checkpoint's 70,300 optimizer steps over 100 epochs. The old 18,652-row cutoff is inconsistent with this evidence.

Local outputs are in `data/reconstructed/` and excluded from Git:

| File | Rows | Purpose |
| --- | ---: | --- |
| `combined_train_validation.csv` | 21,192 | Clean pool for a fresh training/validation split |
| `honma_test.csv` | 1,589 | Original test molecules and labels, with canonical SMILES |
| `combined_reconstructed.csv` | 22,781 | Clean pool and test set together, with split labels |
| `historical_candidate.csv` | 24,108 | Archival candidate retaining the original script's behaviour |

The clean version excludes six missing labels, two invalid SMILES, and 1,325 public rows that duplicate test molecules after canonicalization. Its training and test sets have no exact canonical-molecule overlap. Test identities, order, and labels match the independent `Honma_Ames_Test.xlsx` workbook. The six missing labels also have no matching structure in the older raw Honma training workbook; they were not inferred.

Use `combined_train_validation.csv` for training and `honma_test.csv` for evaluation. The combined and historical files include test rows; the historical file also retains two invalid structures and the original train/test overlap.

`excluded_rows.csv` records each excluded row and reason. `manifest.json` records source/output hashes, counts, labels, and RDKit version. `constructor_verification.json` records the original-script and checkpoint checks. Input row numbers are one-based data rows, excluding headers.

Reproduce from the repository root:

```sh
uv run --no-sync python data_cleaning/reconstruct_combined.py --honma '/mnt/d/Backup/University/5th Year/ames_graphormer/data/raw/Honma_New.xlsx' --test-reference '/mnt/d/Backup/University/5th Year/ames_graphormer/data/raw/Honma_Ames_Test.xlsx'
```

The default public input is `data/raw/Combined.xlsx`, preserving its 10,385 retained rows. The publication CSV removes eight additional ISSSTY rows and is not interchangeable with that workbook.
