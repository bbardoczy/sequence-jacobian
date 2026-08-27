"""Test the public fake_news_matrix on heterogeneous-agent blocks"""

import numpy as np
import pytest

from sequence_jacobian.blocks.het_block import HetBlock


def _check_against_jacobian(block, ss, inputs, outputs, T):
    """J_from_F applied to F must reproduce the block's Jacobian exactly.

    This is the whole contract: F is the intermediate the Jacobian is built
    from, so accumulating it along diagonals has to give the Jacobian back.
    """
    F = block.fake_news_matrix(ss, inputs, outputs, T=T)
    J = block.jacobian(ss, inputs, outputs, T=T)

    assert set(F.keys()) == set(J.outputs)
    for o in J.outputs:
        assert set(F[o].keys()) == set(inputs)
        for i in inputs:
            assert F[o][i].shape == (T, T)
            assert np.allclose(HetBlock.J_from_F(F[o][i]), J[o][i])


def test_het_block_fake_news(krusell_smith_dag):
    _, ss, ks_model, _, _, _ = krusell_smith_dag
    household = ks_model['hh']

    _check_against_jacobian(household, ss, ['r', 'w'], ['A', 'C'], T=10)


def test_fake_news_is_not_the_jacobian(krusell_smith_dag):
    """Guard against fake_news_matrix quietly returning J instead of F."""
    _, ss, ks_model, _, _, _ = krusell_smith_dag
    household = ks_model['hh']

    F = household.fake_news_matrix(ss, ['r'], ['A'], T=10)
    J = household.jacobian(ss, ['r'], ['A'], T=10)

    assert not np.allclose(F['A']['r'], J['A']['r'])
    # They agree on the first row, which J_from_F leaves untouched.
    assert np.allclose(F['A']['r'][0], J['A']['r'][0])


def test_stage_block_fake_news():
    """StageBlock carries its own copy of the algorithm, so check it separately."""
    from .test_stage_block import hh2

    calibration = {'r': 0.004, 'eis': 0.5, 'rho_e': 0.91, 'sd_e': 0.92, 'nE': 3,
                   'amin': 0.0, 'amax': 200, 'nA': 100, 'transfer': 0.143, 'N': 1,
                   'atw': 1, 'beta': 0.97, 'shift': 0}
    ss = hh2.steady_state(calibration)

    _check_against_jacobian(hh2, ss, ['r', 'atw'], ['A', 'C'], T=10)
