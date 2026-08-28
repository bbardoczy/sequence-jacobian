"""Test block symbols, the input to the winding number criterion"""

import numpy as np
import pytest

import sequence_jacobian as sj
from sequence_jacobian.determinacy import unit_circle
from sequence_jacobian.classes.steady_state_dict import SteadyStateDict


@sj.simple
def shifts(x):
    lag = x(-1)
    lead = x(+1)
    same = x
    combo = 2 * x(-1) - 3 * x(+2)
    none = x.ss
    return lag, lead, same, combo, none


@pytest.fixture
def ss():
    return SteadyStateDict({"x": 1.0, "lag": 1.0, "lead": 1.0, "same": 1.0,
                            "combo": -1.0, "none": 1.0})


def test_simple_block_symbol_is_exact(ss):
    """The symbol of a shift is a power of z, with the lag operator giving z."""
    z = unit_circle(64)
    S = shifts._symbol(ss, ["x"], ["lag", "lead", "same", "combo", "none"],
                      T=10, n_sample=64)

    assert np.allclose(S["lag"]["x"], z)
    assert np.allclose(S["lead"]["x"], 1 / z)
    assert np.allclose(S["same"]["x"], 1.0)
    assert np.allclose(S["combo"]["x"], 2 * z - 3 / z**2)
    # An output that ignores its input is a genuine zero, not a missing entry.
    assert np.allclose(S["none"]["x"], 0.0)


def test_symbol_at_one_is_the_cumulative_response(ss):
    """j(1) sums the coefficients: the response to a permanent unit change."""
    S = shifts._symbol(ss, ["x"], ["combo"], T=10, n_sample=64)

    # z[0] is 1, and the coefficients of combo are 2 and -3
    assert np.isclose(S["combo"]["x"][0], 2 - 3)


def test_symbol_matches_the_jacobian_diagonals(ss):
    """Cross-check against reading coefficients off the truncated matrix.

    Legitimate here only because a simple block is exactly banded, so its
    corner is free of the truncation artifacts that rule this out in general.
    """
    T, n = 20, 64
    z = unit_circle(n)
    S = shifts._symbol(ss, ["x"], ["combo"], T=T, n_sample=n)

    M = shifts.jacobian(ss, ["x"], ["combo"], T=T)["combo"]["x"].matrix(T)
    coefs = np.concatenate((M[:, -1], M[-1, :-1][::-1]))     # powers -(T-1)..T-1
    powers = np.arange(-(T - 1), T)
    direct = np.array([np.sum(coefs * zz ** powers) for zz in z])

    assert np.allclose(S["combo"]["x"], direct)


def _corner_coefficients(J):
    """Appendix A.1: read the coefficients off the last column and row of J."""
    T = J.shape[0]
    return np.concatenate((J[:, -1], J[-1, :-1][::-1]))


def test_laurent_from_fake_news_matches_the_jacobian_corner(krusell_smith_dag):
    """Summing F's diagonals equals reading J's corner, exactly.

    J accumulates F along diagonals, so its last row and column already hold the
    complete sum over each diagonal of the truncated F. The two routes therefore
    agree to machine precision; what A.1 approximates is F itself being cut off
    at T, which neither route can see.
    """
    from sequence_jacobian.blocks.het_block import HetBlock
    from sequence_jacobian.determinacy import laurent_from_fake_news

    _, ss, ks_model, _, _, _ = krusell_smith_dag
    hh = ks_model['hh']

    T = 20
    F = hh.fake_news_matrix(ss, ['r'], ['A'], T=T)['A']['r']

    assert np.allclose(laurent_from_fake_news(F),
                       _corner_coefficients(HetBlock.J_from_F(F)))


def test_het_block_symbol(krusell_smith_dag):
    from sequence_jacobian.determinacy import (evaluate_laurent,
                                               laurent_from_fake_news)

    _, ss, ks_model, _, _, _ = krusell_smith_dag
    hh = ks_model['hh']

    T, n = 20, 128
    S = hh._symbol(ss, ['r', 'w'], ['A', 'C'], T=T, n_sample=n)

    assert set(S.keys()) == {'A', 'C'}
    for o in ('A', 'C'):
        for i in ('r', 'w'):
            assert S[o][i].shape == (n,)

    # matches evaluating the coefficients directly
    F = hh.fake_news_matrix(ss, ['r'], ['A'], T=T)['A']['r']
    assert np.allclose(S['A']['r'],
                       evaluate_laurent(laurent_from_fake_news(F), n))

    # j(1) is the sum of the coefficients: the cumulative response
    assert np.isclose(S['A']['r'][0], laurent_from_fake_news(F).sum())


def test_stage_block_symbol():
    from .test_stage_block import hh2

    calibration = {'r': 0.004, 'eis': 0.5, 'rho_e': 0.91, 'sd_e': 0.92, 'nE': 3,
                   'amin': 0.0, 'amax': 200, 'nA': 100, 'transfer': 0.143, 'N': 1,
                   'atw': 1, 'beta': 0.97, 'shift': 0}
    ss = hh2.steady_state(calibration)

    S = hh2._symbol(ss, ['r'], ['A'], T=20, n_sample=128)
    assert S['A']['r'].shape == (128,)
    assert np.isfinite(S['A']['r']).all()


def test_aliasing_is_refused():
    from sequence_jacobian.determinacy import evaluate_laurent

    with pytest.raises(ValueError, match="alias"):
        evaluate_laurent(np.zeros(2 * 20 - 1), n_sample=16)
