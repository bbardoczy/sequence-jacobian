"""Test the winding number criterion for determinacy"""

import warnings

import numpy as np
import pytest

import sequence_jacobian as sj
from sequence_jacobian.classes.steady_state_dict import SteadyStateDict
from sequence_jacobian.determinacy import WindingNumber


def test_taylor_principle(one_asset_hank_dag):
    """The textbook answer, on a shipped model: determinate iff phi > 1."""
    _, ss, model, unknowns, targets, _ = one_asset_hank_dag
    ss = ss.copy()

    for phi, expected in [(0.9, -1), (1.01, 0), (1.5, 0)]:
        ss['phi'] = phi
        assert model.winding_number(ss, unknowns, targets, T=200,
                                    n_sample=2048) == expected, phi


def test_krusell_smith_is_determinate(krusell_smith_dag):
    _, ss, model, unknowns, targets, _ = krusell_smith_dag
    assert model.winding_number(ss, unknowns, targets, T=200, n_sample=2048) == 0


def test_return_path(krusell_smith_dag):
    _, ss, model, unknowns, targets, _ = krusell_smith_dag

    wind, path = model.winding_number(ss, unknowns, targets, T=100,
                                      n_sample=512, return_path=True)
    assert path.shape == (512,)
    assert np.iscomplexobj(path)
    # A determinate model's det keeps well clear of the origin.
    assert np.abs(path).min() / np.abs(path).max() > 1e-3
    assert wind == 0


def test_winding_number_repr():
    assert "unique bounded solution" in repr(WindingNumber(0))
    assert "1-dimensional family" in repr(WindingNumber(-1))
    assert "2-dimensional family" in repr(WindingNumber(-2))
    assert "no bounded solution" in repr(WindingNumber(1))
    # still an int in every way that matters
    assert WindingNumber(-1) == -1
    assert WindingNumber(0) + 1 == 1
    assert isinstance(WindingNumber(0), int)


def test_non_square_system_is_refused(krusell_smith_dag):
    _, ss, model, unknowns, targets, _ = krusell_smith_dag

    with pytest.raises(ValueError, match="not square"):
        model.winding_number(ss, list(unknowns) + ['Z'], targets, T=50,
                             n_sample=256)


def test_solved_block_has_no_symbol():
    """A solved block must be promoted; reaching its symbol is a mistake."""
    @sj.solved(unknowns={'p': (0.5, 2.0)}, targets=['p_resid'])
    def pricing(p, x, beta):
        p_resid = beta * p(+1) + x - p
        return p_resid

    with pytest.raises(NotImplementedError, match="must be promoted"):
        pricing._symbol(SteadyStateDict({}), ['x'], ['p_resid'], T=10,
                        n_sample=64)


def test_promotion_matches_writing_the_model_flat():
    """Promoting a solved block gives the system you would have written by hand."""
    @sj.solved(unknowns={'p': (0.5, 2.0)}, targets=['p_resid'])
    def pricing(p, x, beta):
        p_resid = beta * p(+1) + x - p
        return p_resid

    @sj.simple
    def pricing_flat(p, x, beta):
        p_resid = beta * p(+1) + x - p
        return p_resid

    @sj.simple
    def clearing(p, u):
        resid = p - 2 * u
        return resid

    ss = SteadyStateDict({'x': 1.0, 'beta': 0.95, 'p': 20.0, 'u': 10.0,
                          'p_resid': 0.0, 'resid': 0.0})

    nested = sj.create_model([pricing, clearing], name='nested')
    flat = sj.create_model([pricing_flat, clearing], name='flat')

    a = nested.winding_number(ss, ['u'], ['resid'], T=50, n_sample=512)
    b = flat.winding_number(ss, ['u', 'p'], ['resid', 'p_resid'], T=50,
                            n_sample=512)
    assert a == b


def test_near_zero_determinant_warns():
    """A root just off the circle must be flagged, not reported as a clean 0.

    The symbol here is 1 - (1-eps)z, whose only root sits just outside the unit
    circle, so the winding number is 0 -- but det passes within 1e-10 of the
    origin at z = 1, and a hair's change in the model would flip the verdict.
    That is exactly the situation the criterion cannot speak to.
    """
    @sj.simple
    def almost_singular(u):
        resid = u - 0.9999999999 * u(-1)
        return resid

    ss = SteadyStateDict({'u': 1.0, 'resid': 0.0})
    model = sj.create_model([almost_singular], name='almost')

    with pytest.warns(UserWarning, match="within"):
        wind = model.winding_number(ss, ['u'], ['resid'], T=20, n_sample=256)
    assert wind == 0


def test_identically_singular_system_raises():
    """det j(z) vanishing everywhere has no winding number at all."""
    @sj.simple
    def no_dependence(x, u):
        resid = x + 0 * u          # resid does not move with the unknown
        return resid

    ss = SteadyStateDict({'x': 1.0, 'u': 1.0, 'resid': 1.0})
    model = sj.create_model([no_dependence], name='degenerate')

    with pytest.raises(ValueError, match="vanishes identically"):
        model.winding_number(ss, ['u'], ['resid'], T=20, n_sample=256)


@pytest.mark.parametrize("coefficient, expected", [
    (0.9999999999, "unit root"),          # symbol 1 - a*z, near-root at z = 1
    (-0.9999999999, "period two"),        # symbol 1 + a*z, near-root at z = -1
])
def test_near_zero_says_where(coefficient, expected):
    """The location of the closest approach diagnoses the kind of near-root."""
    @sj.simple
    def near_root(u, a):
        resid = u - a * u(-1)
        return resid

    ss = SteadyStateDict({'u': 1.0, 'a': coefficient, 'resid': 0.0})
    model = sj.create_model([near_root], name='nearroot')

    with pytest.warns(UserWarning, match=expected):
        model.winding_number(ss, ['u'], ['resid'], T=20, n_sample=4096)


def test_near_zero_reports_a_cycle():
    """A complex near-root is reported as an oscillation, with its period."""
    @sj.simple
    def cyclical(u, a):
        resid = u - a * u(-1) + a * u(-2)
        return resid

    # symbol 1 - a z + a z**2 has complex roots just off the circle
    ss = SteadyStateDict({'u': 1.0, 'a': 0.9999999999, 'resid': 0.0})
    model = sj.create_model([cyclical], name='cycle')

    with pytest.warns(UserWarning, match="oscillating with period"):
        model.winding_number(ss, ['u'], ['resid'], T=20, n_sample=16384)
