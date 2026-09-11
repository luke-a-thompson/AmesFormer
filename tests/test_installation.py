"""Integration checks for the compiler-free install and bundled checkpoints."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from click.testing import CliRunner

from graphormer.cli import inference
from graphormer.config.hparams import HyperparameterConfig
from graphormer.predict import predict_smiles
from graphormer.utils import resolve_device


ROOT = Path(__file__).resolve().parents[1]


def test_numpy_torch_interop():
    values = np.array([1.0, 2.0], dtype=np.float32)
    np.testing.assert_array_equal(torch.from_numpy(values).numpy(), values)


@pytest.mark.parametrize("name", ["AmesFormer-Pro", "AmesFormer-Honma"])
def test_pretrained_cpu_predictions(name):
    checkpoint = ROOT / "pretrained_models" / f"{name}.pt"
    results = predict_smiles(["CCO", "c1ccccc1", "[Na+]"], checkpoint, device="cpu")
    assert [result["smiles"] for result in results] == ["CCO", "c1ccccc1", "[Na+]"]
    assert all(np.isfinite(result["probability"]) and 0 <= result["probability"] <= 1 for result in results)


def test_checkpoint_keeps_requested_device():
    config = HyperparameterConfig(
        name="AmesFormer-Pro",
        checkpoint_dir=str(ROOT / "pretrained_models"),
        torch_device="cpu",
        dataset="custom-data.csv",
        batch_size=2,
        num_workers=0,
    )
    config.load_for_inference(verbose=False)
    assert config.torch_device == "cpu"
    assert config.name == "AmesFormer-Pro"
    assert config.dataset == "custom-data.csv"
    assert config.batch_size == 2
    assert config.num_workers == 0


def test_auto_device_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"
    assert resolve_device("cuda:1") == "cuda:1"


def test_inference_checkpoint_cli():
    result = CliRunner().invoke(inference, ["--smiles", "CCO", "--torch_device", "cpu", "--checkpoint", str(ROOT / "pretrained_models/AmesFormer-Pro.pt")])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)[0]["smiles"] == "CCO"


def test_inference_smiles_cli():
    result = CliRunner().invoke(inference, ["--smiles", "CCO", "--smiles", "c1ccccc1", "--name", "AmesFormer-Pro", "--torch_device", "cpu"])
    assert result.exit_code == 0, result.output
    predictions = json.loads(result.output)
    assert [row["smiles"] for row in predictions] == ["CCO", "c1ccccc1"]
    assert all(0 <= row["probability"] <= 1 for row in predictions)


@pytest.mark.parametrize("options", [["--smiles", "CCO"], ["--dataset", "data/raw/Combined.xlsx"]])
def test_inference_requires_explicit_model(options):
    # The name in default_hparams.toml must not silently choose an inference model.
    result = CliRunner().invoke(inference, options)
    assert result.exit_code == 2
    assert "Missing option --name" in result.output


def test_inference_uses_requested_model(monkeypatch, tmp_path):
    from graphormer import cli

    captured = {}
    monkeypatch.setattr(cli, "_predict_json", lambda smiles, checkpoint, *args: captured.update(checkpoint=checkpoint))
    result = CliRunner().invoke(
        inference, ["--smiles", "CCO", "--name", "my-model", "--checkpoint_dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert captured["checkpoint"] == tmp_path / "my-model.pt"


@pytest.mark.parametrize("options", [["--dataset", "data/raw/Combined.xlsx"], ["--mc_samples", "2"]])
def test_inference_rejects_mixed_modes(options):
    result = CliRunner().invoke(inference, ["--smiles", "CCO", "--name", "AmesFormer-Pro", *options])
    assert result.exit_code == 2


def test_inference_dataset_route(monkeypatch):
    from graphormer import cli
    from graphormer.config.options import DatasetRegime

    monkeypatch.setattr(HyperparameterConfig, "load_for_inference", lambda self: self)
    captured = {}

    def evaluate(config, **kwargs):
        captured.update(config=config, options=kwargs)
        return {"labels": [1], "logits": [0.9]}

    monkeypatch.setattr(cli, "inference_model", evaluate)
    monkeypatch.setattr(cli, "save_results", lambda *args: captured.update(saved=args))
    result = CliRunner().invoke(inference, ["--dataset", "data/raw/Combined.xlsx", "--name", "example", "--mc_samples", "2"])
    assert result.exit_code == 0, result.output
    assert captured["config"].dataset == "data/raw/Combined.xlsx"
    assert captured["config"].dataset_regime == DatasetRegime.TEST
    assert captured["options"]["mc_samples"] == 2
    assert captured["saved"] == ({"labels": [1], "logits": [0.9]}, "example", 2)


@pytest.mark.parametrize("smiles", [[], [""], ["invalid"]])
def test_invalid_smiles(smiles):
    with pytest.raises(ValueError):
        predict_smiles(smiles)


@pytest.mark.parametrize("name", ["AmesFormer-Pro", "AmesFormer-Honma"])
def test_predictions_do_not_depend_on_batch_size(name):
    smiles = ["CCO", "c1ccccc1", "CN", "[Na+]", "CC(=O)O"]
    checkpoint = ROOT / "pretrained_models" / f"{name}.pt"
    reference = [row["probability"] for row in predict_smiles(smiles, checkpoint, "cpu", batch_size=1)]
    for batch_size in (2, 3, 5):
        batched = [row["probability"] for row in predict_smiles(smiles, checkpoint, "cpu", batch_size=batch_size)]
        np.testing.assert_allclose(batched, reference, rtol=1e-4, atol=2e-6)


@pytest.mark.parametrize("options", [["--mc_samples", "0"], ["--batch_size", "0"], ["--max_path_distance", "65"]])
def test_inference_rejects_invalid_sizes(options):
    result = CliRunner().invoke(inference, ["--smiles", "CCO", "--name", "AmesFormer-Pro", *options])
    assert result.exit_code == 2
