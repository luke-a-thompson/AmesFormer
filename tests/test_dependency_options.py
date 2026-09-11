"""Ensure the locked environment exposes independent CUDA and Rust switches."""

from pathlib import Path
import shutil
import subprocess

from packaging.markers import default_environment
from packaging.requirements import Requirement
import pytest


@pytest.mark.parametrize("cuda", [False, True])
@pytest.mark.parametrize("rust", [False, True])
def test_environment_options(cuda, rust):
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required to check project dependency selection")
    command = [uv, "export", "--locked", "--offline", "--no-hashes", "--no-header", "--no-annotate"]
    for enabled, extra in [(cuda, "cuda"), (rust, "rust")]:
        if enabled:
            command.extend(["--extra", extra])
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True)
    lines = result.stdout.splitlines()
    environment = default_environment() | {
        "sys_platform": "linux", "platform_machine": "x86_64", "platform_python_implementation": "CPython"
    }
    torch = [Requirement(line) for line in lines if line.startswith("torch==")]
    selected = [str(requirement.specifier) for requirement in torch if not requirement.marker or requirement.marker.evaluate(environment)]
    assert selected == ["==2.3.1+cu121" if cuda else "==2.3.1+cpu"]
    assert any(line.startswith("nvidia-") for line in lines) == cuda
    assert ("./gnn-tools" in lines) == rust
