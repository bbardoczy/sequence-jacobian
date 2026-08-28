"""Class definition of a simple block"""

import numpy as np
from copy import deepcopy

from .support.simple_displacement import ignore, Displace, AccumulatedDerivative
from .block import Block
from ..classes import SteadyStateDict, ImpulseDict, JacobianDict, SimpleSparse
from ..utilities import misc
from ..utilities.function import ExtendedFunction
from ..determinacy import unit_circle

'''Part 1: SimpleBlock class and @simple decorator to generate it'''


def simple(f):
    return SimpleBlock(f)


class SimpleBlock(Block):
    """Generated from simple block written in Dynare-ish style and decorated with @simple, e.g.

    @simple
    def production(Z, K, L, alpha):
        Y = Z * K(-1) ** alpha * L ** (1 - alpha)
        return Y

    which is a SimpleBlock that takes in Z, K, L, and alpha, all of which can be either constants
    or series, and implements a Cobb-Douglas production function, noting that for production today
    we use the capital K(-1) determined yesterday.

    Key methods are .ss, .td, and .jac, like HetBlock.
    """

    def __init__(self, f):
        super().__init__()
        self.f = ExtendedFunction(f)
        self.name = self.f.name
        self.inputs = self.f.inputs
        self.outputs = self.f.outputs

    def __repr__(self):
        return f"<SimpleBlock '{self.name}'>"

    def _steady_state(self, ss):
        outputs = self.f.wrapped_call(ss, preprocess=ignore, postprocess=misc.numeric_primitive)
        return SteadyStateDict({**ss, **outputs})

    def _impulse_nonlinear(self, ss, inputs, outputs, ss_initial):
        if ss_initial is None:
            ss_initial = ss
            ss_initial_flag = False
        else:
            ss_initial_flag = True

        input_args = {}
        for k, v in inputs.items():
            if np.isscalar(v):
                raise ValueError(f'Keyword argument {k}={v} is scalar, should be time path.')
            input_args[k] = Displace(v + ss[k], ss[k], ss_initial[k], k)

        for k in self.inputs:
            if k not in input_args:
                if not ss_initial_flag or (ss_initial_flag and np.array_equal(ss_initial[k], ss[k])):
                    input_args[k] = ignore(ss[k])
                else:
                    input_args[k] = Displace(np.full(inputs.T, ss[k]), ss[k], ss_initial[k], k)

        return ImpulseDict(make_impulse_uniform_length(self.f(input_args)))[outputs] - ss

    def _impulse_linear(self, ss, inputs, outputs, Js):
        return ImpulseDict(self.jacobian(ss, list(inputs.keys()), outputs, inputs.T, Js).apply(inputs))

    def _jacobian(self, ss, inputs, outputs, T):
        invertedJ = {i: {} for i in inputs}

        # Loop over all inputs/shocks which we want to differentiate with respect to
        for i in inputs:
            invertedJ[i] = self.compute_single_shock_J(ss, i)

        # Because we computed the Jacobian of all outputs with respect to each shock (invertedJ[i][o]),
        # we need to loop back through to have J[o][i] to map for a given output `o`, shock `i`,
        # the Jacobian curlyJ^{o,i}.
        J = {o: {} for o in outputs}
        for o in outputs:
            for i in inputs:
                # drop zeros from JacobianDict
                if invertedJ[i][o] and not invertedJ[i][o].iszero:
                    J[o][i] = invertedJ[i][o]

        return JacobianDict(J, outputs, inputs, self.name, T)

    def _symbol(self, ss, inputs, outputs, T, n_sample):
        """Symbol of this block's Jacobian, on the n_sample-th roots of unity.

        Exact, and needs no truncation. Differentiating a simple block gives a
        SimpleSparse, whose elements (i, m) -> x mean x on the diagonal i above
        the main one, with its first m entries missing. Only the diagonal index
        carries the Toeplitz part: m describes a finite-rank correction near the
        boundary, which is compact and does not enter the symbol. So the symbol
        is simply sum of x * z**(-i) over the elements.

        Sign convention: i > 0 sits above the diagonal and is therefore a lead,
        contributing z**(-i). The lag operator accordingly has symbol z, as in
        Auclert, Majic, Rognlie and Straub (2025).

        An absent entry is a genuine zero -- _jacobian drops entries that
        differentiate to zero -- so every requested pair is filled in, whether
        or not the Jacobian has it.

        T is unused: a SimpleSparse does not depend on it. It stays in the
        signature so that every block type answers to the same call.
        """
        z = unit_circle(n_sample)
        J = self.jacobian(ss, inputs, outputs, T=T)

        symbols = {}
        for o in outputs:
            row = {}
            for i in inputs:
                s = np.zeros_like(z)
                x = J[o].get(i) if o in J.outputs else None
                if x is not None:
                    for (diag, _missing), coef in x.elements.items():
                        s = s + coef * z ** (-diag)
                row[i] = s
            symbols[o] = row
        return symbols

    def compute_single_shock_J(self, ss, i):
        input_args = {i: ignore(ss[i]) for i in self.inputs}
        input_args[i] = AccumulatedDerivative(f_value=ss[i])

        J = {o: {} for o in self.outputs}
        for o_name, o in self.f(input_args).items():
            if isinstance(o, AccumulatedDerivative):
                J[o_name] = SimpleSparse(o.elements)

        return J


# TODO: move this to impulse.py?
def make_impulse_uniform_length(out):
    T = np.max([np.size(v) for v in out.values()])
    return {k: (np.full(T, misc.numeric_primitive(v)) if np.isscalar(v) else misc.numeric_primitive(v))
                                                        for k, v in out.items()}
