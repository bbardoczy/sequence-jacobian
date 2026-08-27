"""Test SimpleSparse arithmetic, including the identically-zero operator"""

import numpy as np
import pytest

from sequence_jacobian.classes.jacobian_dict import JacobianDict
from sequence_jacobian.classes.sparse_jacobians import SimpleSparse

# An operator with no elements is not a pathological input. __add__ deletes
# elements that cancel, so a sum that cancels exactly produces one, as happens
# in a government budget where interest paid on debt nets against interest
# received on the same debt.
CANCELLED = SimpleSparse({(0, 0): 1.0, (1, 0): 0.5}) + SimpleSparse(
    {(0, 0): -1.0, (1, 0): -0.5}
)
EXPLICIT_ZERO = SimpleSparse({(0, 0): 0.0})


def test_cancellation_empties_elements():
    assert CANCELLED.elements == {}
    assert CANCELLED.iszero
    # An explicit zero coefficient is also zero, but keeps its element.
    assert EXPLICIT_ZERO.elements == {(0, 0): 0.0}
    assert EXPLICIT_ZERO.iszero


@pytest.mark.parametrize("z", [CANCELLED, EXPLICIT_ZERO])
def test_zero_operator_arithmetic(z):
    """Both representations of zero must behave the same in every product."""
    T = 4
    A = np.arange(T * T, dtype=float).reshape(T, T)
    v = np.arange(T, dtype=float)

    assert np.allclose(z.matrix(T), 0.0)
    assert np.allclose(z @ A, 0.0)
    assert np.allclose(A @ z, 0.0)
    assert np.allclose(z @ v, 0.0)
    assert (z @ SimpleSparse({(1, 0): 2.0})).iszero


def test_zero_operator_array_is_jittable():
    """Empty arrays must keep the rank and dtype multiply_rs_matrix compiled for."""
    indices, xs = CANCELLED.array()
    nonempty_indices, nonempty_xs = SimpleSparse({(1, 0): 2.0}).array()

    assert indices.shape == (0, 2)
    assert xs.shape == (0,)
    assert indices.dtype == nonempty_indices.dtype
    assert xs.dtype == nonempty_xs.dtype


def test_zero_operator_composes_with_dense_jacobian():
    """The failure this guards against: composing a cancelled operator downstream.

    A cancelled SimpleSparse reaching a dense Jacobian -- as it does whenever a
    heterogeneous-agent block sits downstream of an exactly cancelling
    accounting identity -- used to raise ValueError inside compose.
    """
    T = 4
    dense = JacobianDict({"out": {"mid": np.eye(T)}}, T=T)
    sparse = JacobianDict({"mid": {"in": CANCELLED}}, T=T)

    composed = dense @ sparse

    assert np.allclose(composed["out"]["in"] @ np.eye(T), 0.0)
