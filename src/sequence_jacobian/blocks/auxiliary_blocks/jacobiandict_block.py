"""A simple wrapper for JacobianDicts to be embedded in DAGs"""

import numpy as np

from ..block import Block
from ...classes import ImpulseDict, JacobianDict
from ...classes.sparse_jacobians import IdentityMatrix, SimpleSparse
from ...determinacy import (evaluate_laurent, laurent_from_matrix,
                            unit_circle)

class JacobianDictBlock(JacobianDict, Block):
    """A wrapper for nested dicts/JacobianDicts passed directly into DAGs to ensure method compatibility"""
    def __init__(self, nesteddict, outputs=None, inputs=None, name=None):
        # A block inside a DAG is addressed by name: it is the key of `kids`, of
        # `descendants`, and of any `Js` supplied to reuse a Jacobian. Two
        # anonymous blocks are therefore indistinguishable, so refuse to make one
        # rather than let it collide later, somewhere less obviously connected.
        if name is None and getattr(nesteddict, 'name', None) in (None, 'NestedDict'):
            raise ValueError(
                'JacobianDictBlock requires a name. Pass a named JacobianDict, or '
                'wrap it yourself: JacobianDictBlock(J, name="household").')
        super().__init__(nesteddict, outputs=outputs, inputs=inputs, name=name)
        Block.__init__(self)

    def __repr__(self):
        return f"<JacobianDictBlock outputs={self.outputs}, inputs={self.inputs}>"

    def _impulse_linear(self, ss, inputs, outputs, Js):
        return ImpulseDict(self.jacobian(ss, list(inputs.keys()), outputs, inputs.T, Js).apply(inputs))

    def _jacobian(self, ss, inputs, outputs, T):
        if not inputs <= self.inputs:
            raise KeyError(f'Asking JacobianDictBlock for {inputs - self.inputs}, which are among its inputs {self.inputs}')
        if not outputs <= self.outputs:
            raise KeyError(f'Asking JacobianDictBlock for {outputs - self.outputs}, which are among its outputs {self.outputs}')
        return self[outputs, inputs]

    def _symbol(self, ss, inputs, outputs, T, n_sample):
        """Symbol of the stored Jacobian, evaluated on the roots of unity.

        Coefficients come off the corner of each matrix, which is sound only if
        that corner is representative -- see determinacy.laurent_from_matrix.
        A Jacobian handed to this block carries no record of where it came from,
        so this cannot be checked here.

        In practice: safe for a Jacobian written by the fake news algorithm,
        which is the usual reason to precompute one. Not safe for a Jacobian
        that is itself a composition, where truncation artifacts sit exactly
        where the coefficients are read.

        Nor for a solved block's Jacobian, which is a truncated inverse. At best
        its corner approximates the symbol the implicit function theorem would
        give, which already drops the solved block's own winding number (see
        Block.winding_number). When that winding number is not zero, the corner
        is not even that: it grows geometrically, is set by the boundary
        condition the truncation imposes, and can bring the count out at
        -(T-1) with no warning. Put the solved block itself in the model
        instead, so that winding_number can promote it.
        """
        symbols = {}
        for o in outputs:
            row = {}
            for i in inputs:
                x = self[o].get(i) if o in self.outputs else None
                if x is None:
                    row[i] = np.zeros(n_sample, dtype=complex)
                elif isinstance(x, IdentityMatrix):
                    row[i] = np.ones(n_sample, dtype=complex)
                elif isinstance(x, SimpleSparse):
                    z = unit_circle(n_sample)
                    acc = np.zeros(n_sample, dtype=complex)
                    for (diag, _missing), coef in x.elements.items():
                        acc = acc + coef * z ** (-diag)
                    row[i] = acc
                else:
                    row[i] = evaluate_laurent(laurent_from_matrix(np.asarray(x)),
                                              n_sample)
            symbols[o] = row
        return symbols
