"""Train a fresh Pro-sized model; select on validation and evaluate the untouched test set."""
import argparse
import csv
import hashlib
import json
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from graphormer.config.hparams import HyperparameterConfig
from graphormer.config.utils import calculate_pos_weight, save_checkpoint
from graphormer.data.dataloader import GraphormerDataLoader
from graphormer.data.datasets.graphormer_dataset import FileDataset
from graphormer.train import Trainer


class NullWriter:
    def add_scalar(self, *args, **kwargs):
        pass


def metrics(labels, probabilities):
    predictions = probabilities >= 0.5
    return {
        'accuracy': accuracy_score(labels, predictions),
        'balanced_accuracy': balanced_accuracy_score(labels, predictions),
        'roc_auc': roc_auc_score(labels, probabilities),
        'pr_auc': average_precision_score(labels, probabilities),
        'log_loss': log_loss(labels, probabilities),
        'brier': brier_score_loss(labels, probabilities),
        'confusion_matrix': confusion_matrix(labels, predictions).tolist(),
    }


@torch.inference_mode()
def evaluate(model, loader, loss_function, device):
    model.eval()
    total, count, logits, labels = 0.0, 0, [], []
    for batch in loader:
        batch = batch.to(device)
        output = model(batch)
        total += loss_function(output, batch.y).item() * batch.y.numel()
        count += batch.y.numel()
        logits.append(output.cpu())
        labels.append(batch.y.cpu())
    return total / count, torch.cat(labels).numpy(), torch.cat(logits).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/reconstructed/training_run'))
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_float32_matmul_precision('high')
    device = torch.device(args.device)
    template = torch.load('pretrained_models/AmesFormer-Pro.pt', map_location='cpu', weights_only=False)
    known = {field.name for field in fields(HyperparameterConfig)}
    hp = HyperparameterConfig(**{key: value for key, value in template['hyperparameters'].items() if key in known})
    hp.name = 'AmesFormer-Pro-reconstructed'
    hp.torch_device = str(device)
    hp.dataset = 'data/reconstructed/combined_train_validation.csv'
    hp.test_size = 0.8
    hp.epochs = args.epochs
    hp.start_epoch = 0
    hp.best_loss = float('inf')
    hp.checkpoint_dir = str(args.output)
    hp.num_workers = 4
    hp.prefetch_factor = 2
    # Constructor state dictionaries are empty: no pretrained weights are reused.
    del template
    prep_start = time.perf_counter()
    dataset = FileDataset(hp.dataset, max_distance=hp.max_path_distance)
    test_path = Path('data/reconstructed/honma_test.csv')
    test_dataset = FileDataset(test_path, max_distance=hp.max_path_distance)
    valid_ids, train_ids = train_test_split(range(len(dataset)), test_size=hp.test_size, random_state=42)
    (args.output / 'split_indices.json').write_text(json.dumps({'train': train_ids, 'validation': valid_ids}) + '\n')
    loader_kwargs = dict(batch_size=hp.batch_size, num_workers=4, persistent_workers=True, prefetch_factor=2, pin_memory=True)
    train_loader = GraphormerDataLoader(Subset(dataset, train_ids), shuffle=True, **loader_kwargs)
    valid_loader = GraphormerDataLoader(Subset(dataset, valid_ids), **loader_kwargs)
    test_loader = GraphormerDataLoader(test_dataset, **loader_kwargs)
    model = hp.model_config().build(node_feature_dim=dataset.num_node_features, edge_feature_dim=dataset.num_edge_features, output_dim=1).to(device)
    optimizer = hp.optimizer_config().build(model)
    scheduler = hp.scheduler_config().build(optimizer)
    loss_function = hp.loss_config().with_pos_weight(calculate_pos_weight(train_loader)).with_weights(hp.loss_weights).build()
    trainer = Trainer(hp, scheduler, model, train_loader, valid_loader, device, optimizer, loss_function, NullWriter(), hp.batch_size)
    prep_seconds = time.perf_counter() - prep_start
    settings = {'seed': 42, 'torch': torch.__version__, 'device': torch.cuda.get_device_name() if device.type == 'cuda' else str(device),
                'train_rows': len(train_ids), 'validation_rows': len(valid_ids), 'test_rows': len(test_dataset),
                'batch_size': hp.batch_size, 'max_epochs': args.epochs, 'patience': args.patience, 'learning_rate': hp.lr,
                'precision': 'float32, matmul_precision=high', 'preprocessing_seconds': prep_seconds,
                'train_sha256': hashlib.sha256(Path(hp.dataset).read_bytes()).hexdigest(),
                'test_sha256': hashlib.sha256(test_path.read_bytes()).hexdigest()}
    (args.output / 'settings.json').write_text(json.dumps(settings, indent=2) + '\n')
    print(json.dumps(settings), flush=True)
    best_epoch = -1
    history = []
    train_start = time.perf_counter()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    with (args.output / 'history.csv').open('w', buffering=1) as stream:
        writer = csv.DictWriter(stream, fieldnames=['epoch', 'train_loss', 'validation_loss', 'validation_auc', 'validation_balanced_accuracy', 'seconds'])
        writer.writeheader()
        for epoch in range(args.epochs):
            start = time.perf_counter()
            model.train()
            optimizer.zero_grad()
            trainer._accumulated_samples = 0
            total, samples = 0.0, 0
            losses = []
            for index, batch in enumerate(train_loader):
                batch_loss = trainer.train_step(batch, index, epoch*len(train_loader)+index, losses, len(train_loader))
                total += batch_loss * batch.y.numel()
                samples += batch.y.numel()
            validation_loss, labels, logits = evaluate(model, valid_loader, loss_function, device)
            probabilities = torch.sigmoid(torch.from_numpy(logits)).numpy()
            row = {'epoch': epoch+1, 'train_loss': total/samples, 'validation_loss': validation_loss,
                   'validation_auc': roc_auc_score(labels, probabilities),
                   'validation_balanced_accuracy': balanced_accuracy_score(labels, probabilities >= .5),
                   'seconds': time.perf_counter()-start}
            history.append(row)
            writer.writerow(row)
            print(json.dumps(row), flush=True)
            if validation_loss < hp.best_loss:
                hp.best_loss = validation_loss
                best_epoch = epoch
                save_checkpoint(epoch, hp, model, optimizer, loss_function, scheduler, 'best')
            if epoch - best_epoch >= args.patience:
                print(f'Early stopping after {epoch+1} epochs; best epoch {best_epoch+1}', flush=True)
                break
    training_seconds = time.perf_counter()-train_start
    best_path = args.output / f'{hp.name}_best.pt'
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best['model_state_dict'])
    test_start = time.perf_counter()
    _, labels, logits = evaluate(model, test_loader, loss_function, device)
    test_seconds = time.perf_counter()-test_start
    probabilities = torch.sigmoid(torch.from_numpy(logits)).numpy()
    frame = pd.read_csv(test_path)[['smiles', 'ames']]
    frame['probability'] = probabilities
    frame.to_csv(args.output / 'retrained_test_predictions.csv', index=False)
    result = {'settings': settings, 'epochs_completed': len(history), 'best_epoch': best_epoch+1,
              'best_validation_loss': hp.best_loss, 'training_seconds': training_seconds,
              'median_epoch_seconds': float(np.median([row['seconds'] for row in history])),
              'test_inference_seconds': test_seconds, 'metrics': metrics(labels, probabilities),
              'checkpoint': str(best_path), 'peak_gpu_memory_gb': torch.cuda.max_memory_allocated()/1e9 if device.type == 'cuda' else None}
    (args.output / 'retrained_result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
