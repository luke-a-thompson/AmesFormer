"""Summarize the fixed test-set comparison after train_reconstructed.py finishes."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def main():
    root = Path('data/reconstructed/training_run')
    result = json.loads((root/'retrained_result.json').read_text())
    frame = pd.read_csv(root/'retrained_test_predictions.csv')
    labels = frame.ames.to_numpy()
    models = {name: json.loads((root/f'{name}.json').read_text()) for name in ['pro_legacy', 'pro_corrected']}
    probabilities = {name: np.asarray(model['probabilities']) for name, model in models.items()}
    probabilities['retrained'] = frame.probability.to_numpy()
    models['retrained'] = result
    rng = np.random.default_rng(42)
    groups = [np.flatnonzero(labels == value) for value in [0, 1]]
    differences = {name: {'roc_auc': [], 'balanced_accuracy': []} for name in ['pro_legacy', 'pro_corrected']}
    for _ in range(1000):
        indices = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups])
        y = labels[indices]
        fresh = probabilities['retrained'][indices]
        fresh_auc = roc_auc_score(y, fresh)
        fresh_bac = balanced_accuracy_score(y, fresh >= .5)
        for name in differences:
            old = probabilities[name][indices]
            differences[name]['roc_auc'].append(fresh_auc - roc_auc_score(y, old))
            differences[name]['balanced_accuracy'].append(fresh_bac - balanced_accuracy_score(y, old >= .5))
    intervals = {name: {metric: np.quantile(values, [.025, .975]).tolist() for metric, values in stats.items()} for name, stats in differences.items()}
    (root/'paired_bootstrap.json').write_text(json.dumps(intervals, indent=2)+'\n')
    table = []
    display = {'pro_legacy': 'Bundled Pro, previous code', 'pro_corrected': 'Bundled Pro, corrected code', 'retrained': 'Freshly trained, corrected code'}
    for name, model in models.items():
        m = model['metrics']
        table.append(f"| {display[name]} | {m['accuracy']:.1%} | {m['balanced_accuracy']:.1%} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} | {m['log_loss']:.4f} |")
    uncertainty = []
    for name, stats in intervals.items():
        auc_low, auc_high = stats['roc_auc']
        bac_low, bac_high = stats['balanced_accuracy']
        uncertainty.append(f"- Retrained minus {display[name].lower()}: ROC-AUC difference 95% interval [{auc_low:+.4f}, {auc_high:+.4f}]; balanced accuracy difference [{100*bac_low:+.1f}, {100*bac_high:+.1f}] percentage points.")
    report = f'''# Retraining on reconstructed Combined data

A fresh Pro-sized model was trained on the reconstructed pool, with seed 42 and an 80/20 training/validation split: 16,954 training, 4,238 validation, and 1,589 separate Honma test molecules. No exact canonical molecules overlap between the training/validation pool and test set. The test set was evaluated after checkpoint selection; early stopping and checkpoint selection used weighted validation BCE only.

| Model | Accuracy | Balanced accuracy | ROC-AUC | PR-AUC | Log loss |
| --- | ---: | ---: | ---: | ---: | ---: |
'''+'\n'.join(table)+f'''

- Same Pro architecture: three layers, hidden/edge dimensions 128, FFN dimension 80, four heads, dropout 0.05. Fresh random weights; AdamW at 0.00015, batch size 32, constant learning rate, class-weighted BCE, gradient clipping at 5.
- Maximum 100 epochs; stop after 15 epochs without lower validation loss. Completed {result['epochs_completed']} epochs; selected epoch {result['best_epoch']}.
- GPU: {result['settings']['device']}, PyTorch {result['settings']['torch']}, float32 with high matmul precision. Training took {result['training_seconds']/60:.1f} minutes; median epoch {result['median_epoch_seconds']:.1f} seconds; preprocessing {result['settings']['preprocessing_seconds']:.1f} seconds. Test evaluation took {result['test_inference_seconds']:.2f} seconds. Peak allocated GPU memory: {result['peak_gpu_memory_gb']:.2f} GB.
- Accuracy and balanced accuracy use threshold 0.5. No temperature or threshold was fitted on test labels. PR-AUC is average precision.
- The previous-code baseline uses commit 911ec3f and original test SMILES spelling; corrected models use canonical SMILES. Existing Pro weights were not modified. All three were evaluated with the same CUDA PyTorch runtime.
- The bundled Pro's historical training candidate contains 1,325 canonical overlaps with this test set. The freshly trained model excludes them. Different training data, validation fraction, and one random seed limit causal claims about the model-code fixes alone.

Local artifacts: `data/reconstructed/training_run/` contains the best checkpoint, split indices, data hashes, epoch history, per-molecule test predictions, baseline probabilities, and paired bootstrap intervals. Bootstrap intervals describe uncertainty over test molecules, not variation across training seeds.

'''+ '\n'.join(uncertainty)+'''

The RTX 5080 run used the existing CUDA PyTorch installation under `/home/luke/sig-benchmarks/.venv` through a temporary module overlay, with the AmesFormer environment providing other dependencies. The repository's pinned PyTorch 2.3.1 environment was not changed; it does not support this GPU architecture.
'''
    Path('reports/retraining-results.md').write_text(report)
    print(report)
    print('Paired bootstrap 95% intervals (retrained minus baseline):',json.dumps(intervals))


if __name__ == '__main__':
    main()
