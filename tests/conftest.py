import pytest
import torch


@pytest.fixture(autouse=True)
def restore_grad_mode():
    """Keep tests that disable autograd from affecting subsequent training tests."""
    enabled = torch.is_grad_enabled()
    yield
    torch.set_grad_enabled(enabled)
