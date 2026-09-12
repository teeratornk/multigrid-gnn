import numpy as np
import pytest
import torch

torch.set_default_dtype(torch.float64)


@pytest.fixture
def rng():
    """Function scoped on purpose. A session-scoped generator carries state
    between tests, so results depend on how many ran first and on which
    subset was selected. That made a borderline fp32 assertion pass under
    `pytest` and fail under `pytest tests/test_dtype_device.py`."""
    return np.random.default_rng(0)


@pytest.fixture(params=[16, 32], ids=lambda n: f"n{n}")
def cavity(request):
    from gnn4buoyancy import build_case
    return build_case("conjugate_cavity", n=request.param)


@pytest.fixture
def sink():
    from gnn4buoyancy import build_case
    return build_case("heat_sink", n=32, fins=4, contrast=1e3)


@pytest.fixture
def bore():
    from gnn4buoyancy import build_case
    return build_case("engine_bore", n=32, contrast=1e3)
