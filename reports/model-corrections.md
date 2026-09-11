# Model correction comparison

Baseline: commit `911ec3f`. This comparison used the original Pro and Honma weights unchanged; Pro has since been replaced by the [retrained checkpoint](retraining-results.md). CPU, PyTorch 2.3.1, batch size 16. Inputs: eight diagnostic SMILES followed by the first 128 rows of `data/raw/Combined.xlsx` (136 inputs per model). These are prediction comparisons, not held-out accuracy estimates.

| Model | Refactor-only maximum change | Corrected mean absolute change | Corrected maximum change |
| --- | ---: | ---: | ---: |
| AmesFormer-Pro | 8e-07 | 0.3103 | 0.9536 |
| AmesFormer-Honma | 9.2e-07 | 0.2464 | 0.9387 |

[All predictions](model-predictions.csv). Changes are absolute probabilities, not percentages.

- Pro originally scored ethanol as 0.234373, 0.298798, and 0.201423 for `CCO`, `OCC`, and `C(O)C`; corrected scores all agree at 0.942659 within floating-point precision.
- Fixed graph/node/edge indexing and physical atom degrees; graphs now contain exactly one path per ordered node pair. Cache version increases to v3.
- MHA now uses per-head scaling and dropout on attention weights through PyTorch SDPA. FISH and linear attention, their configuration flags, and FISH diagnostics/tests are removed.
- Calibration caches one checkpoint's validation logits. Training averages gradients over actual accumulated samples before clipping, saves checkpoints by sample-averaged validation loss, and counts incomplete accumulation windows in OneCycleLR. Both inference modes respect explicit temperature overrides.
- Configuration dataclasses replace repeated assignments; retained constructor fields were compared against the original classes. Tuning configuration shrinks from 604 to 389 lines, including removal of obsolete attention settings. Model configuration shrinks from 281 to 90 lines.
- Edge encoding projects edges before gathering paths. Double-precision output and gradient differences were below 2e-14. A float32 CPU microbenchmark with two threads and B=8, N=40, E=120, L=5, D=64 measured 6.43 ms before and 0.22 ms after (28.9× for this operation). The gathered intermediate has 64× fewer elements. This is not an end-to-end or GPU speed claim.

Validation: 87 tests pass with the compiled Rust backend; 83 pass and four Rust parity tests skip with Python only. Tests cover graph paths, equivalent SMILES, batching, attention math/gradients, edge-encoding equivalence, cached calibration, gradient accumulation, best-checkpoint selection, scheduler steps, CLI temperature parity, and actual training/inference. CUDA execution was not tested in this initial comparison; the subsequent retraining report covers the GPU run.

The numerical fixes substantially change predictions from the old weights. The subsequent Pro retraining addresses this; Honma still requires revalidation. Published accuracy does not carry over automatically. New checkpoints record model_version=2.
