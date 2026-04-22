import pytest
import torch

@pytest.fixture
def sample_input():
    return torch.randn(1, 3, 32, 32)
