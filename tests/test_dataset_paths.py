import pandas as pd
import pytest
from click.testing import CliRunner

from graphormer.cli import inference, train
from graphormer.config.data import DataConfig
from graphormer.config.options import DatasetRegime
from graphormer.data.datasets.graphormer_dataset import FileDataset


@pytest.fixture
def table():
    return pd.DataFrame({"smiles": ["CCO", "CC", "CCC", "CO", "CN", "CCN", "C", "CNC"], "ames": [0, 1] * 4})


@pytest.mark.parametrize("extension", ["csv", "xlsx"])
def test_direct_file_loading_and_splits(tmp_path, table, extension):
    path = tmp_path / f"my molecules.{extension}"
    getattr(table, f"to_{'excel' if extension == 'xlsx' else 'csv'}")(path, index=False)
    config = DataConfig(str(path), DatasetRegime.TRAIN, 2, 3).with_num_workers(0).with_test_size(0.5).with_random_state(42)
    training, validation = config.build()
    assert len(training.dataset) == len(validation.dataset) == 4
    assert set(training.dataset.indices).isdisjoint(validation.dataset.indices)
    assert next(iter(training)).y.numel() == 2
    config.dataset_regime = DatasetRegime.TEST
    assert len(config.build().dataset) == 8


def test_cache_tracks_contents_and_path_distance(tmp_path, table):
    path = tmp_path / "molecules.csv"
    table.to_csv(path, index=False)
    first = FileDataset(path, max_distance=3)
    longer = FileDataset(path, max_distance=5)
    assert first.processed_paths != longer.processed_paths
    table.iloc[:4].to_csv(path, index=False)
    changed = FileDataset(path, max_distance=3)
    assert len(changed) == 4
    assert first.processed_paths != changed.processed_paths


@pytest.mark.parametrize("name,start", [("Combined", 18652), ("Honma_New", 12140), ("Hansen_New", 6513), ("tox24_all", 1014)])
def test_benchmark_split_is_preserved(tmp_path, table, name, start):
    path = tmp_path / f"{name}.csv"
    table.to_csv(path, index=False)
    config = DataConfig(str(path), DatasetRegime.TEST, 2, 3).with_num_workers(0).with_test_size(0.8).with_random_state(42)
    with pytest.raises(ValueError, match=f"starts at row {start}"):
        config.build()


def test_missing_columns(tmp_path):
    path = tmp_path / "molecules.csv"
    path.write_text("smiles\nCCO\n")
    with pytest.raises(ValueError, match="missing: ames"):
        FileDataset(path)


def test_train_and_inference_with_file(tmp_path, table, monkeypatch):
    from graphormer import cli

    path = tmp_path / "molecules.csv"
    table.to_csv(path, index=False)
    runner = CliRunner()
    result = runner.invoke(train, [
        "--dataset", str(path), "--name", "path-test", "--checkpoint_dir", str(tmp_path / "models"),
        "--logdir", str(tmp_path / "runs"), "--torch_device", "cpu", "--num_workers", "0",
        "--epochs", "1", "--batch_size", "3", "--test_size", "0.5",
        "--num_layers", "1", "--hidden_dim", "16", "--ffn_hidden_dim", "16",
    ])
    assert result.exit_code == 0, result.output + repr(result.exception)
    captured = {}
    monkeypatch.setattr(cli, "save_results", lambda results, *args: captured.update(results=results))
    result = runner.invoke(inference, [
        "--dataset", str(path), "--name", "AmesFormer-Pro", "--torch_device", "cpu", "--num_workers", "0",
        "--batch_size", "7",
    ])
    assert result.exit_code == 0, result.output + repr(result.exception)
    assert len(captured["results"]["labels"]) == len(table)


@pytest.mark.parametrize("command", [train, inference])
def test_invalid_dataset_path(command, tmp_path):
    result = CliRunner().invoke(command, ["--dataset", str(tmp_path / "missing.csv")])
    assert result.exit_code == 2
    assert "does not exist" in result.output


@pytest.mark.parametrize("label", [2, -1, None, "invalid"])
def test_invalid_labels_are_rejected(tmp_path, label):
    path = tmp_path / "molecules.csv"
    pd.DataFrame({"smiles": ["CCO"], "ames": [label]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="binary labels"):
        FileDataset(path)


@pytest.mark.parametrize("smiles", ["invalid", "", None])
def test_invalid_smiles_are_rejected(tmp_path, smiles):
    path = tmp_path / "Combined.csv"
    pd.DataFrame({"smiles": [smiles, "CCO"], "ames": [0, 1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="SMILES"):
        FileDataset(path)


def test_mc_samples_have_independent_results(tmp_path, table, monkeypatch):
    from graphormer import cli

    path = tmp_path / "molecules.csv"
    table.to_csv(path, index=False)
    captured = {}
    monkeypatch.setattr(cli, "save_results", lambda results, *args: captured.update(results=results))
    result = CliRunner().invoke(inference, [
        "--dataset", str(path), "--name", "AmesFormer-Pro", "--num_workers", "0",
        "--torch_device", "cpu", "--batch_size", "7", "--mc_samples", "2",
    ])
    assert result.exit_code == 0, result.output + repr(result.exception)
    samples = captured["results"]
    assert len(samples) == 2
    assert samples[0]["labels"] is not samples[1]["labels"]
    assert samples[0]["logits"] is not samples[1]["logits"]
    for sample in samples:
        assert sample["labels"] == table.ames.tolist()
        assert len(sample["logits"]) == len(table)
