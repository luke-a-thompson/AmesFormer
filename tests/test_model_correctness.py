"""Regression checks for graph semantics, calibration, and optimizer updates."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_smiles

from graphormer.config.hparams import HyperparameterConfig
from graphormer.config.options import LossReductionType, OptimizerType
from graphormer.config.optimizer import OptimizerConfig
from graphormer.data.data_cleaning import process
from graphormer.data.dataloader import GraphormerDataLoader
from graphormer.modules.attention import GraphormerMultiHeadAttention
from graphormer.modules.encoding.edge_encoding import EdgeEncoding
from graphormer.optimise_temperature import TemperatureOptimizer
from graphormer.predict import predict_smiles
from graphormer.train import Trainer


@pytest.mark.parametrize('smiles', ['CCO', '[Na+]', 'C.C', 'c1ccccc1', 'CC(=O)O'])
def test_graph_indices_and_paths(smiles):
    original = from_smiles(smiles)
    g = process(original.clone(), 8)
    n = g.num_nodes
    paths = g.node_paths.reshape(n, n, 8)
    edges = g.edge_paths.reshape(n, n, 8)
    assert len(g.node_paths) == n*n
    assert g.edge_index.shape[1] == len(g.edge_attr)
    torch.testing.assert_close(g.degrees[1:, 0], torch.bincount(original.edge_index[0], minlength=n-1).float())
    for source in range(n):
        assert paths[source, source].tolist() == [source] + [-1]*7
        for target in range(n):
            nodes = paths[source, target][paths[source, target] >= 0].tolist()
            edge_ids = edges[source, target][edges[source, target] >= 0].tolist()
            if not nodes:
                assert not edge_ids
                continue
            assert (nodes[0], nodes[-1]) == (source, target)
            assert len(edge_ids) == len(nodes)-1
            for left, right, edge in zip(nodes, nodes[1:], edge_ids):
                assert set(g.edge_index[:, edge].tolist()) == {left, right}
    if smiles == 'C.C':
        assert (paths[1, 2] == -1).all()


@pytest.mark.parametrize('name', ['AmesFormer-Pro', 'AmesFormer-Honma'])
def test_equivalent_smiles_have_equal_predictions(name):
    result = predict_smiles(['CCO', 'OCC', 'C(O)C'], f'pretrained_models/{name}.pt', 'cpu')
    np.testing.assert_allclose([r['probability'] for r in result], result[0]['probability'], atol=2e-6, rtol=0)


def test_attention_matches_reference_and_has_finite_gradients():
    torch.manual_seed(71)
    attention = GraphormerMultiHeadAttention(2, 8, 0).double()
    x = torch.randn(2, 4, 8, dtype=torch.double, requires_grad=True)
    padding = torch.tensor([[False, False, False, False], [False, True, True, True]])
    prior = torch.randn(2, 1, 4, 4, dtype=torch.double, requires_grad=True)
    data = SimpleNamespace(normalized_input=x, attention_prior=prior, x_pad_mask=padding)
    actual = attention(data).attention_output
    q, k, v = [p(x).reshape(2, 4, 2, 4).transpose(1, 2) for p in (attention.linear_q, attention.linear_k, attention.linear_v)]
    scores = (q @ k.transpose(-1, -2)) / 2 + prior
    weights = scores.masked_fill(padding[:, None, None, :], -torch.inf).softmax(-1)
    reference = attention.linear_out((weights @ v).transpose(1, 2).reshape(2, 4, 8)).masked_fill(padding[..., None], 0)
    torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)
    actual.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(prior.grad).all()


def test_mc_dropout_reaches_attention():
    attention = GraphormerMultiHeadAttention(2, 8, 0.5).eval()
    attention.att_dropout.train()
    x = torch.randn(1, 4, 8)
    data = SimpleNamespace(normalized_input=x, attention_prior=torch.zeros(1, 1, 4, 4), x_pad_mask=None)
    first = attention(data).attention_output.clone()
    assert not torch.equal(first, attention(data).attention_output)


def test_edge_projection_preserves_outputs_and_gradients():
    torch.manual_seed(81)
    embeddings = torch.randn(2, 12, 6, dtype=torch.double, requires_grad=True)
    paths = torch.randint(-2, 14, (2, 7, 7, 5))
    encoding = EdgeEncoding(6, 5).double()
    torch.nn.init.normal_(encoding.edge_vector)
    actual = encoding(SimpleNamespace(edge_paths=paths, edge_embedding=embeddings)).edge_encoding
    invalid = (paths < 0) | (paths >= 12)
    gathered = embeddings[torch.arange(2)[:, None, None, None], paths.masked_fill(invalid, 0)]
    gathered = gathered.masked_fill(invalid[..., None], 0)
    reference = torch.einsum('bnmld,ld->bnm', gathered, encoding.edge_vector) / ((~invalid).sum(-1).float() + encoding.eps)
    torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)
    grads = [torch.autograd.grad(value.square().sum(), (embeddings, encoding.edge_vector), retain_graph=True) for value in (actual, reference)]
    for actual_grad, reference_grad in zip(*grads):
        torch.testing.assert_close(actual_grad, reference_grad, atol=1e-12, rtol=1e-12)


def test_temperature_uses_one_checkpoint_forward_pass(monkeypatch):
    hp = HyperparameterConfig(name='AmesFormer-Pro', torch_device='cpu')
    hp.load_for_inference(verbose=False)
    graphs = [process(from_smiles(s), hp.max_path_distance) for s in ['CCO', 'CC', 'N']]
    for i, graph in enumerate(graphs):
        graph.y = torch.tensor([i % 2], dtype=torch.float)
    loader = GraphormerDataLoader(graphs, batch_size=2)
    dc = SimpleNamespace(num_node_features=9, num_edge_features=3, build=lambda: (None, loader))
    monkeypatch.setattr(hp, 'data_config', lambda: dc)
    from graphormer.modules.model import Graphormer
    original_forward = Graphormer.forward
    calls = []
    def count_forward(self, batch):
        calls.append(1)
        return original_forward(self, batch)
    monkeypatch.setattr(Graphormer, 'forward', count_forward)
    calibration = TemperatureOptimizer(hp)
    assert len(calls) == 2
    labels, predictions = calibration.get_predictions(1)
    np.testing.assert_array_equal(labels, [0, 1, 0])
    np.testing.assert_array_equal(predictions, calibration.get_predictions(1)[1])
    np.testing.assert_allclose(calibration.get_predictions(2)[1], torch.sigmoid(torch.tensor(calibration.logits)/2).numpy(), atol=1e-7)
    assert len(calls) == 2
    # A configuration can build the same checkpoint repeatedly.
    config = hp.model_config()
    first = config.build(node_feature_dim=9, edge_feature_dim=3, output_dim=1)
    second = config.build()
    for left, right in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(left, right)


class RegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1).double()
    def forward(self, batch):
        return self.linear(batch.x).squeeze(-1)


@pytest.mark.parametrize('reduction', [LossReductionType.MEAN, LossReductionType.SUM])
@pytest.mark.parametrize('samples', [5, 7])
def test_accumulation_matches_actual_large_batches(reduction, samples):
    torch.manual_seed(52)
    model = RegressionModel()
    reference = deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01)
    hp = SimpleNamespace(accumulation_steps=3, clip_grad_norm=0.1, loss_reduction=reduction)
    trainer = Trainer(hp, None, model, None, None, torch.device('cpu'), optimizer, torch.nn.MSELoss(reduction=reduction.value), Mock(), 6)
    x, y = torch.randn(samples, 2, dtype=torch.double), torch.randn(samples, dtype=torch.double)
    batches = [Data(x=x[i:i+2], y=y[i:i+2]) for i in range(0, samples, 2)]
    for i, batch in enumerate(batches):
        trainer.train_step(batch, i, i, [], len(batches))
    for i in range(0, samples, 6):
        batch = Data(x=x[i:i+6], y=y[i:i+6])
        torch.nn.functional.mse_loss(reference(batch), batch.y).backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.1)
        ref_optimizer.step()
        ref_optimizer.zero_grad()
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_optimizer_lr_is_independent_of_accumulation():
    config = OptimizerConfig(OptimizerType.ADAMW, LossReductionType.MEAN, 4, 8, 0.01)
    optimizer = config.with_betas((0.9, 0.999)).with_eps(1e-8).with_weight_decay(0).build(RegressionModel())
    assert optimizer.param_groups[0]['lr'] == 0.01


def test_best_checkpoint_uses_sample_average(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from graphormer.cli import train
    import graphormer.train as training
    path = tmp_path / 'molecules.csv'
    path.write_text('smiles,ames\n' + 'CCO,0\nCC,1\n'*4)
    saved = []
    def evaluate(self, batch, index, predictions, labels):
        predictions.extend([0]*len(batch.y))
        labels.extend(batch.y.tolist())
        return [0.6, 0.2, 0.4, 0.0][index]
    def save(epoch, hp, *args):
        if args[-1] == 'best':
            saved.append((epoch, hp.best_loss))
    monkeypatch.setattr(Trainer, 'eval_step', evaluate)
    monkeypatch.setattr(training, 'save_checkpoint', save)
    result = CliRunner().invoke(train, [
        '--dataset', str(path), '--name', 'checkpoint-test', '--checkpoint_dir', str(tmp_path / 'models'),
        '--logdir', str(tmp_path / 'runs'), '--torch_device', 'cpu', '--num_workers', '0',
        '--epochs', '2', '--batch_size', '3', '--test_size', '0.5',
        '--num_layers', '1', '--hidden_dim', '16', '--ffn_hidden_dim', '16',
    ])
    assert result.exit_code == 0, result.output + repr(result.exception)
    np.testing.assert_allclose(saved, [(0, 0.5), (1, 0.3)])


def test_one_cycle_counts_incomplete_accumulation_windows():
    from graphormer.config.scheduler import SchedulerConfig
    from graphormer.config.options import SchedulerType
    optimizer = torch.optim.SGD(RegressionModel().parameters(), lr=0.01)
    config = SchedulerConfig(SchedulerType.ONE_CYCLE, 2, 3)
    for key, value in dict(max_lr=0.1, train_batches_per_epoch=5, total_iters=2, pct_start=0.3,
                           anneal_strategy='cos', cycle_momentum=False, base_momentum=0.85,
                           max_momentum=0.95, div_factor=25, final_div_factor=10000,
                           three_phase=False, last_effective_batch_num=-1).items():
        setattr(config, key, value)
    scheduler = config.build(optimizer)
    assert scheduler.total_steps == 6
    assert optimizer.param_groups[0]['max_lr'] == 0.1
    for _ in range(6):
        optimizer.step()
        scheduler.step()


def test_cli_temperature_matches_in_both_inference_modes(tmp_path, monkeypatch):
    import json
    from click.testing import CliRunner
    from graphormer import cli
    path = tmp_path / 'molecules.csv'
    path.write_text('smiles,ames\nCCO,0\nCC,1\n')
    common = ['--name', 'AmesFormer-Pro', '--torch_device', 'cpu', '--num_workers', '0', '--temperature', '2']
    result = CliRunner().invoke(cli.inference, ['--smiles', 'CCO', '--smiles', 'CC', *common])
    assert result.exit_code == 0, result.output
    smiles = json.loads(result.output)
    captured = {}
    monkeypatch.setattr(cli, 'save_results', lambda results, *args: captured.update(results))
    result = CliRunner().invoke(cli.inference, ['--dataset', str(path), *common])
    assert result.exit_code == 0, result.output
    np.testing.assert_allclose(captured['logits'], [row['probability'] for row in smiles], atol=1e-7)


def test_no_norm_builds_and_removed_attention_flags_fail():
    from click.testing import CliRunner
    from graphormer.cli import inference
    from graphormer.config.options import NormType
    hp = HyperparameterConfig(name='AmesFormer-Pro', torch_device='cpu')
    hp.load_for_inference(verbose=False)
    config = hp.model_config()
    config.norm_type = NormType.NONE
    config.state_dict = None
    model = config.build(node_feature_dim=9, edge_feature_dim=3, output_dim=1).eval()
    batch = next(iter(GraphormerDataLoader([process(from_smiles('CCO'), hp.max_path_distance)])))
    assert torch.isfinite(model(batch)).all()
    for option in ['fish', 'linear']:
        result = CliRunner().invoke(inference, ['--smiles', 'CCO', '--name', 'AmesFormer-Pro', '--attention_type', option])
        assert result.exit_code == 2
