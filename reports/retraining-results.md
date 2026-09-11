# Retraining on reconstructed Combined data

A fresh Pro-sized model was trained on the reconstructed pool, with seed 42 and an 80/20 training/validation split: 16,954 training, 4,238 validation, and 1,589 separate Honma test molecules. No exact canonical molecules overlap between the training/validation pool and test set. The test set was evaluated after checkpoint selection; early stopping and checkpoint selection used weighted validation BCE only.

| Model | Accuracy | Balanced accuracy | ROC-AUC | PR-AUC | Log loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bundled Pro, previous code | 78.5% | 65.3% | 0.7405 | 0.3734 | 0.4916 |
| Bundled Pro, corrected code | 58.7% | 63.0% | 0.6712 | 0.2473 | 1.0150 |
| Freshly trained, corrected code | 80.7% | 72.8% | 0.7863 | 0.4709 | 0.4436 |

- Same Pro architecture: three layers, hidden/edge dimensions 128, FFN dimension 80, four heads, dropout 0.05. Fresh random weights; AdamW at 0.00015, batch size 32, constant learning rate, class-weighted BCE, gradient clipping at 5.
- Maximum 100 epochs; stop after 15 epochs without lower validation loss. Completed 68 epochs; selected epoch 53.
- GPU: NVIDIA GeForce RTX 5080, PyTorch 2.12.0+cu130, float32 with high matmul precision. Training took 17.7 minutes; median epoch 14.8 seconds; preprocessing 38.4 seconds. Test evaluation took 0.98 seconds. Peak allocated GPU memory: 0.48 GB.
- Accuracy and balanced accuracy use threshold 0.5. No temperature or threshold was fitted on test labels. PR-AUC is average precision.
- The previous-code baseline uses commit 911ec3f and original test SMILES spelling; corrected models use canonical SMILES. Existing Pro weights were not modified. All three were evaluated with the same CUDA PyTorch runtime.
- The bundled Pro's historical training candidate contains 1,325 canonical overlaps with this test set. The freshly trained model excludes them. Different training data, validation fraction, and one random seed limit causal claims about the model-code fixes alone.

Local artifacts: `data/reconstructed/training_run/` contains the best checkpoint, split indices, data hashes, epoch history, per-molecule test predictions, baseline probabilities, and paired bootstrap intervals. Bootstrap intervals describe uncertainty over test molecules, not variation across training seeds.

- Retrained minus bundled pro, previous code: ROC-AUC difference 95% interval [+0.0069, +0.0863]; balanced accuracy difference [+3.1, +12.0] percentage points.
- Retrained minus bundled pro, corrected code: ROC-AUC difference 95% interval [+0.0696, +0.1621]; balanced accuracy difference [+5.3, +13.9] percentage points.

The RTX 5080 run used the existing CUDA PyTorch installation under `/home/luke/sig-benchmarks/.venv` through a temporary module overlay, with the AmesFormer environment providing other dependencies. The repository's pinned PyTorch 2.3.1 environment was not changed; it does not support this GPU architecture.
