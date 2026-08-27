"""A simple wrapper for JacobianDicts to be embedded in DAGs"""

from ..block import Block
from ...classes import ImpulseDict, JacobianDict

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
