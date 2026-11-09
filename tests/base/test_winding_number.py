"""Test the winding number criterion for determinacy"""

import warnings

import numpy as np
import pytest

import sequence_jacobian as sj
from sequence_jacobian.blocks.block import Block
from sequence_jacobian.classes.steady_state_dict import SteadyStateDict
from sequence_jacobian.determinacy import WindingNumber, winding_number_of_path


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


@pytest.mark.parametrize('beta, c, expected', [
    (0.5, 0.5, 0),      # pricing determinate on its own, and so is the model
    (0.9, 2.0, -1),     # pricing determinate on its own, the model is not
    (1.5, 2.0, -1),     # pricing indeterminate on its own, and so is the model
    (1.5, 0.2, 0),      # pricing indeterminate on its own, clearing pins it down
])
def test_promotion_matches_writing_the_model_flat(beta, c, expected):
    """Promoting a solved block gives the system you would have written by hand.

    And only promoting keeps the solved block's own winding in the count. pricing
    solves p_t = beta p_{t+1} + u_t, and clearing sets resid_t = p_t - c u_t.
    Written flat, j(z) = [[-c, 1], [1, beta/z - 1]], so det j(z) is
    (c - 1) - c beta / z, whose winding number is -1 when c beta > |c - 1| and
    zero otherwise.

    The tempting alternative is to give the solved block a symbol by the
    implicit function theorem, dp/du = -j_pu / j_pp pointwise in z, and to wind
    the outer system alone. Its symbol is then det j(z) / j_pp(z), so its
    winding number is the model's minus the solved block's own: that of
    beta/z - 1, which is -1 when beta > 1. Nothing warns, since nothing comes
    near zero on the circle. Harmless while pricing is determinate on its own,
    wrong once it is not -- calling the third case unique, and finding no
    solution in the fourth, a model that is fine. (beta = 1/phi reads pricing as
    Fisher plus a Taylor rule, solved for inflation, so beta > 1 is a violated
    Taylor principle hidden inside a solved block.)
    """
    @sj.solved(unknowns={'p': (-10.0, 10.0)}, targets=['p_resid'])
    def pricing(p, u, beta):
        p_resid = beta * p(+1) + u - p
        return p_resid

    @sj.simple
    def pricing_flat(p, u, beta):
        p_resid = beta * p(+1) + u - p
        return p_resid

    @sj.simple
    def clearing(p, u, c):
        resid = p - c * u
        return resid

    # zero is a steady state at every beta and c
    ss = SteadyStateDict({'u': 0.0, 'p': 0.0, 'beta': beta, 'c': c,
                          'p_resid': 0.0, 'resid': 0.0})
    T, n_sample = 50, 512

    nested = sj.create_model([pricing, clearing], name='nested')
    flat = sj.create_model([pricing_flat, clearing], name='flat')

    assert nested.winding_number(ss, ['u'], ['resid'], T=T,
                                 n_sample=n_sample) == expected
    assert flat.winding_number(ss, ['u', 'p'], ['resid', 'p_resid'], T=T,
                               n_sample=n_sample) == expected

    # The implicit function theorem route, from the same two blocks.
    inner = pricing.block._symbol(ss, ['u', 'p'], ['p_resid'], T,
                                  n_sample)['p_resid']
    outer = clearing._symbol(ss, ['u', 'p'], ['resid'], T, n_sample)['resid']
    dp_du = -inner['u'] / inner['p']
    own = winding_number_of_path(inner['p'])
    unpromoted = winding_number_of_path(outer['u'] + outer['p'] * dp_du)

    assert own == (-1 if beta > 1 else 0)
    assert unpromoted == expected - own


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


def test_check_existence_on_a_healthy_model(krusell_smith_dag):
    """A model with a genuine solution does not grow with the horizon."""
    _, ss, model, unknowns, targets, _ = krusell_smith_dag

    growth = model.check_existence(ss, unknowns, targets, T=60, T_compare=120,
                                   n_draws=10)
    assert growth < 2.0


def test_check_existence_detects_a_singular_system():
    """The lag operator is the textbook nonexistence case.

    L maps (x0, x1, ...) to (0, x0, ...): injective, but nothing with a non-zero
    first entry is in its range, so Lx = y usually has no solution. Truncated, it
    is outright singular rather than merely ill-behaved.
    """
    @sj.simple
    def lagged(u):
        resid = u(-1)
        return resid

    ss = SteadyStateDict({'u': 1.0, 'resid': 1.0})
    model = sj.create_model([lagged], name='lag')

    assert model.winding_number(ss, ['u'], ['resid'], T=20, n_sample=256) == 1

    with pytest.warns(UserWarning, match="exactly singular"):
        assert model.check_existence(ss, ['u'], ['resid'], T=20, T_compare=40,
                                     n_draws=5) == np.inf


def test_check_existence_needs_a_rebuildable_model():
    """A precomputed Jacobian has a fixed T, so the probe cannot vary it."""
    from sequence_jacobian.classes.jacobian_dict import JacobianDict

    T = 30
    J = JacobianDict({'C': {'Y': np.tril(np.ones((T, T))) * 0.1}},
                     name='mpcs', T=T)

    @sj.simple
    def goods(Y, C):
        goods_mkt = Y - C
        return goods_mkt

    model = sj.create_model([J, goods], name='fixedT')
    ss = SteadyStateDict({'Y': 1.0, 'C': 1.0, 'goods_mkt': 0.0})

    with pytest.raises(ValueError, match="Cannot rebuild this model"):
        model.check_existence(ss, ['Y'], ['goods_mkt'], T=10, T_compare=20,
                              n_draws=3)


@pytest.mark.parametrize("coefficient, expected", [
    (0.9999999999, "unit root"),          # symbol 1 - a*z, root essentially at z = 1
    (-0.9999999999, "period two"),        # symbol 1 + a*z, root essentially at z = -1
])
def test_root_on_the_circle_warns_and_says_where(coefficient, expected):
    """A root on the unit circle costs us the count, and the guard says so.

    The argument of det j(z) jumps by pi as the path passes through the origin,
    which no amount of sampling resolves, so the turn is what we watch. Where the
    path comes closest then diagnoses the kind of root, which is the useful part:
    z = 1 is a unit root, z = -1 alternates, a complex z is a cycle.
    """
    @sj.simple
    def near_root(u, a):
        resid = u - a * u(-1)
        return resid

    ss = SteadyStateDict({'u': 1.0, 'a': coefficient, 'resid': 0.0})
    model = sj.create_model([near_root], name='nearroot')

    with pytest.warns(UserWarning) as record:
        model.winding_number(ss, ['u'], ['resid'], T=20, n_sample=4096)

    message = str(record[0].message)
    assert "turns by" in message          # the count is at risk
    assert expected in message            # and this is the kind of root


def test_a_model_near_a_boundary_does_not_warn(one_asset_hank_dag):
    """Being near a determinacy boundary is not an error, and must stay quiet.

    det j(z) routinely passes within 1e-3 of the origin relative to its own scale
    in models the criterion classifies correctly -- no further away than genuinely
    broken ones -- so proximity alone is no reason to warn. Only the count being
    at risk is.
    """
    _, ss, model, unknowns, targets, _ = one_asset_hank_dag
    ss = ss.copy()
    ss['phi'] = 2.5          # det passes within 3e-3 of the origin, yet wind is right

    with warnings.catch_warnings():
        warnings.simplefilter('error')          # any warning fails the test
        assert model.winding_number(ss, unknowns, targets, T=200,
                                    n_sample=2048) == 0


def test_the_public_surface_is_the_two_block_methods():
    """Determinacy is reached through blocks, not through the top namespace.

    ``sj.determinacy`` holds the machinery and stays importable, but nothing from
    it belongs in ``sj.*``: users call ``model.winding_number(...)`` and
    ``model.check_existence(...)`` and never construct these objects themselves.
    Pinned because ``sj.determinacy`` would otherwise resolve only as a side
    effect of ``block.py`` importing from it.
    """
    assert sj.determinacy.__name__ == 'sequence_jacobian.determinacy'

    for method in ('winding_number', 'check_existence'):
        assert callable(getattr(Block, method))

    # nothing leaked upward, where it would sit confusingly close to the methods
    for name in ('winding_number', 'WindingNumber', 'winding_number_of_path',
                 'unit_circle', 'evaluate_laurent', 'laurent_from_matrix',
                 'laurent_from_fake_news', 'describe_near_zero'):
        assert not hasattr(sj, name), f'sj.{name} should not be public'


@pytest.mark.parametrize('displacement, a, expected', [
    (+1, 0.5, -1),      # a lead, root inside: indeterminacy of dimension one
    (+1, 2.0, 0),       # a lead, but the constant dominates
    (+2, 0.5, -2),      # two leads
    (+2, 2.0, 0),
    (-1, 0.5, +1),      # a lag, root inside: nonexistence of dimension one
    (-1, 2.0, 0),
    (-2, 0.5, +2),      # two lags
])
def test_a_solved_block_may_displace_someone_elses_unknown(displacement, a,
                                                           expected):
    """A solved block that leads or lags a variable which is not its unknown.

    _promote_solved lifts a solved block's own unknown out, so that case takes
    care of itself. The one worth pinning is the other: the block solves for u
    while displacing x, a global unknown belonging elsewhere, and the symbol has
    to be composed through the promotion correctly.

    Each case is built twice, once with the displacement inside the solved block
    and once written directly as a simple block. They describe the same
    operator, so they must agree with each other and with the closed form.

    With u_t = x_{t+d} and res_t = u_t - a x_t + (a - 1), the symbol is
    z**(-d) - a, on the convention that a lag contributes z and a lead z**(-1).
    On the unit circle |z**(-d)| = 1, so by Rouche the winding number is that of
    z**(-d) when a < 1, namely -d, and that of the constant -a when a > 1,
    namely zero. Covering both signs and a dimension above one would catch a
    sign error, an off-by-one, or a collapse to dimension one.
    """
    d = displacement

    @sj.solved(unknowns={'u': (-10.0, 10.0)}, targets=['res_u'])
    def displace(u, x):
        res_u = x(d) - u
        return res_u

    @sj.simple
    def target(u, x, a):
        res_x = u - a * x + (a - 1)
        return res_x

    @sj.simple
    def direct(x, a):
        res_x = x(d) - a * x + (a - 1)
        return res_x

    ss = SteadyStateDict({'x': 1.0, 'u': 1.0, 'a': a,
                          'res_u': 0.0, 'res_x': 0.0})

    # A model may not share a name with a block it contains.
    via_solved = sj.create_model([displace, target], name='model_via_solved')
    plain = sj.create_model([direct], name='model_direct')

    assert via_solved.winding_number(ss, ['x'], ['res_x'], T=100,
                                     n_sample=2048) == expected
    assert plain.winding_number(ss, ['x'], ['res_x'], T=100,
                                n_sample=2048) == expected
