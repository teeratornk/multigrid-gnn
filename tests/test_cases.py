"""The example operators must be what they claim to be."""
import numpy as np
import pytest
import scipy.sparse as sp

from gnn4buoyancy import build_case


ALL_CASES = [("conjugate_cavity", dict(n=16)),
             ("heat_sink", dict(n=16, fins=2, contrast=1e2)),
             ("engine_bore", dict(n=16, contrast=1e2))]


@pytest.mark.parametrize("name,kw", ALL_CASES)
def test_operator_is_square_and_symmetric(name, kw):
    A = build_case(name, **kw)
    assert A.shape[0] == A.shape[1]
    assert abs(A - A.T).max() / abs(A).max() < 1e-12


@pytest.mark.parametrize("name,kw", ALL_CASES)
def test_operator_is_positive_definite(name, kw):
    import scipy.sparse.linalg as spla
    A = build_case(name, **kw)
    lo = spla.eigsh(A.astype(float), k=1, which="SA", return_eigenvectors=False)[0]
    assert lo > 0, f"smallest eigenvalue {lo:.3e}"


def test_heat_sink_actually_has_a_contrast():
    lo = build_case("heat_sink", n=32, fins=4, contrast=1.0)
    hi = build_case("heat_sink", n=32, fins=4, contrast=1e3)
    assert abs(hi).max() / abs(lo).max() > 100, "the fins are not conducting"


def test_engine_bore_has_three_regions():
    """Gas, liner and metal must all be present, otherwise the curved interface
    the case exists for is not there."""
    import numpy as np
    A = build_case("engine_bore", n=64, contrast=1e3, liner_contrast=3e2)
    d = np.unique(np.round(A.diagonal(), 6))
    assert len(d) > 3, f"expected several distinct conductivity levels, got {d[:6]}"


def test_engine_bore_interface_is_curved():
    """A ragged, staircased interface is the point. If the diagonal were constant
    along rows the geometry would be rectangular, not a bore."""
    import numpy as np
    n = 64
    d = build_case("engine_bore", n=n).diagonal().reshape(n, n)
    widths = [(row < row.max() * 0.9).sum() for row in d]
    assert len(set(widths)) > 5, "bore width does not vary down the section"


def test_unknown_case_is_rejected():
    with pytest.raises(KeyError, match="unknown case"):
        build_case("not_a_case")
