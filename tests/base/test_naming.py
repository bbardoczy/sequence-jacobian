"""Test that Jacobians carry the name of the block that produced them"""

import pytest

import sequence_jacobian as sj
from sequence_jacobian.classes.jacobian_dict import JacobianDict
from sequence_jacobian.classes.steady_state_dict import SteadyStateDict


@sj.simple
def hh_inner(a, u):
    m = 2 * a + u
    return m


@sj.simple
def clearing(m, u):
    resid = m - 3 * u
    return resid


@pytest.fixture
def ss():
    return SteadyStateDict({"a": 1.0, "u": 1.0, "m": 3.0, "resid": 0.0})


def test_combined_block_names_its_jacobian(ss):
    """A CombinedBlock names its Jacobian after itself, as leaf blocks do."""
    assert hh_inner.jacobian(ss, ["a"], T=3).name == "hh_inner"

    hh = sj.create_model([hh_inner], name="hh")
    assert hh.jacobian(ss, ["a", "u"], ["m"], T=3).name == "hh"


def test_slicing_preserves_name(ss):
    hh = sj.create_model([hh_inner], name="hh")
    J = hh.jacobian(ss, ["a", "u"], ["m"], T=3)

    assert J[["m"], :].name == "hh"
    assert J[["m"], ["a"]].name == "hh"
    # addinputs() goes through __or__, the other place a new dict is built
    assert J.addinputs().name == "hh"


def test_identity_and_unpack_accept_a_name():
    assert JacobianDict.identity(["a"], name="ident").name == "ident"


def test_embedded_precomputed_jacobian_keeps_its_identity(ss):
    """A precomputed Jacobian embedded in a DAG is addressable by its own name."""
    hh = sj.create_model([hh_inner], name="hh")
    J = hh.jacobian(ss, ["a", "u"], ["m"], T=3)

    outer = sj.create_model([J, clearing], name="outer")
    assert list(outer.kids.keys()) == ["hh", "clearing"]


def test_solve_jacobian_with_embedded_precomputed_jacobian(ss):
    """The regression: anonymous U_Z used to collide with an anonymous kid.

    solve_jacobian combines its internal U_Z into the DAG. When both that and an
    embedded precomputed Jacobian defaulted to "NestedDict", Parent rejected the
    result outright -- so solve_jacobian was unusable on any model built around a
    Jacobian precomputed from a CombinedBlock.
    """
    hh = sj.create_model([hh_inner], name="hh")
    J = hh.jacobian(ss, ["a", "u"], ["m"], T=3)
    outer = sj.create_model([J, clearing], name="outer")

    G = outer.solve_jacobian(ss, ["u"], ["resid"], ["a"], T=3)

    assert "resid" in G.outputs


@sj.simple
def produces_x(a):
    x = a
    return x


@sj.simple
def produces_y(a):
    y = a
    return y


def test_anonymous_jacobian_dict_block_is_refused(ss):
    """Refuse to build a block that cannot be addressed by name."""
    from sequence_jacobian.blocks.auxiliary_blocks.jacobiandict_block import (
        JacobianDictBlock)

    hh = sj.create_model([hh_inner], name="hh")
    J = hh.jacobian(ss, ["a", "u"], ["m"], T=3)

    # A named JacobianDict needs nothing further.
    assert JacobianDictBlock(J).name == "hh"
    assert JacobianDictBlock(J, name="override").name == "override"

    # A raw dict has no name to inherit.
    with pytest.raises(ValueError, match="requires a name"):
        JacobianDictBlock({"m": {"a": J["m"]["a"]}})

    # Nor does create_model, which wraps under the hood.
    with pytest.raises(ValueError, match="requires a name"):
        sj.create_model([{"m": {"a": J["m"]["a"]}}, clearing], name="outer")


def test_duplicate_block_names_are_refused():
    """Two blocks sharing a name used to be silently absorbed into one kid."""
    first = produces_x.rename("dup")
    second = produces_y.rename("dup")

    with pytest.raises(ValueError, match="Overlapping block name dup"):
        sj.create_model([first, second], name="outer")
