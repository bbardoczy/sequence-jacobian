"""CombinedBlock class and the combine function to generate it"""

from .block import Block
from .auxiliary_blocks.jacobiandict_block import JacobianDictBlock
from .support.parent import Parent
import numpy as np

from ..classes import ImpulseDict, JacobianDict
from ..utilities.graph import DAG, find_intermediate_inputs


def combine(blocks, name="", model_alias=False):
    return CombinedBlock(blocks, name=name, model_alias=model_alias)


# Useful functional alias
def create_model(blocks, **kwargs):
    return combine(blocks, model_alias=True, **kwargs)


class CombinedBlock(Block, Parent, DAG):
    """A combined `Block` object comprised of several `Block` objects, which topologically sorts them and provides
    a set of partial and general equilibrium methods for evaluating their steady state, computes impulse responses,
    and calculates Jacobians along the DAG"""
    # To users: Do *not* manually change the attributes via assignment. Instantiating a
    #   CombinedBlock has some automated features that are inferred from initial instantiation but not from
    #   re-assignment of attributes post-instantiation.
    def __init__(self, blocks, name="", model_alias=False, sorted_indices=None, intermediate_inputs=None):
        super().__init__()

        blocks_unsorted = [b if isinstance(b, Block) else JacobianDictBlock(b) for b in blocks]
        DAG.__init__(self, blocks_unsorted)
        
        # TODO: deprecate this, use DAG methods instead
        self._required = find_intermediate_inputs(blocks) if intermediate_inputs is None else intermediate_inputs

        if not name:
            self.name = f"{self.blocks[0].name}_to_{self.blocks[-1].name}_combined"
        else:
            self.name = name

        # now that it has a name, do Parent initialization
        Parent.__init__(self, blocks)

        # If the create_model() is used instead of combine(), we will have __repr__ show this object as a 'Model'
        self._model_alias = model_alias

    def __repr__(self):
        if self._model_alias:
            return f"<Model '{self.name}'>"
        else:
            return f"<CombinedBlock '{self.name}'>"

    def _steady_state(self, calibration, dissolve, **kwargs):
        """Evaluate a partial equilibrium steady state of the CombinedBlock given a `calibration`"""

        ss = calibration.copy()
        for block in self.blocks:
            # TODO: make this inner_dissolve better, clumsy way to dispatch dissolve only to correct children
            inner_dissolve = [k for k in dissolve if self.descendants[k] == block.name]
            outputs = block.steady_state(ss, dissolve=inner_dissolve, **kwargs)
            ss.update(outputs)

        return ss

    def _impulse_nonlinear(self, ss, inputs, outputs, internals, Js, options, ss_initial):
        original_outputs = outputs
        outputs = (outputs | self._required) - ss._vector_valued()

        impulses = inputs.copy()
        for block in self.blocks:
            input_args = {k: v for k, v in impulses.items() if k in block.inputs}

            if input_args or ss_initial is not None:
                # If this block is actually perturbed, or we start from different initial ss
                # TODO: be more selective about ss_initial here - did any inputs change that matter for this one block?
                impulses.update(block.impulse_nonlinear(ss, input_args, outputs & block.outputs, internals, Js, options, ss_initial))

        return ImpulseDict({k: impulses.toplevel[k] for k in original_outputs if k in impulses.toplevel}, impulses.internals, impulses.T)

    def _impulse_linear(self, ss, inputs, outputs, Js, options):
        original_outputs = outputs
        outputs = (outputs | self._required) - ss._vector_valued()
        
        impulses = inputs.copy()
        for block in self.blocks:
            input_args = {k: v for k, v in impulses.items() if k in block.inputs} 

            if input_args:  # If this block is actually perturbed
                impulses.update(block.impulse_linear(ss, input_args, outputs & block.outputs, Js, options))

        return ImpulseDict({k: impulses.toplevel[k] for k in original_outputs if k in impulses.toplevel}, T=impulses.T)

    def _partial_jacobians(self, ss, inputs, outputs, T, Js, options):
        vector_valued = ss._vector_valued()
        inputs = (inputs | self._required) - vector_valued
        outputs = (outputs | self._required) - vector_valued

        curlyJs = {}
        for block in self.blocks:
            curlyJ = block.partial_jacobians(ss, inputs & block.inputs, outputs & block.outputs, T, Js, options)
            curlyJs.update(curlyJ)
            
        return curlyJs

    def _symbol(self, ss, inputs, outputs, T, n_sample):
        """Symbol of this DAG, composed from its blocks rather than from its Jacobian.

        Walks the blocks in evaluation order -- DAG already sorted them -- and
        carries d v / d inputs for every variable produced so far, as one array
        of values on the unit circle per input. Composing a block is then a
        pointwise multiply-add per sampled z, instead of a T x T matrix product.

        Composing symbols rather than reading one off the composed Jacobian is
        the whole point: a composed Jacobian carries truncation artifacts in
        exactly the corner the coefficients would be read from.
        """
        ones = np.ones(n_sample, dtype=complex)
        zeros = np.zeros(n_sample, dtype=complex)

        # Seed with the identity, so an input maps to itself.
        total = {i: {i: ones} for i in inputs}

        vector_valued = ss._vector_valued()
        inner_inputs = (inputs | self._required) - vector_valued
        inner_outputs = (outputs | self._required) - vector_valued

        for block in self.blocks:
            if not ((inner_inputs & block.inputs) and (inner_outputs & block.outputs)):
                continue

            block_outputs = inner_outputs & block.outputs
            block_inputs = inner_inputs & block.inputs
            S = block._symbol(ss, block_inputs, block_outputs, T, n_sample)

            # Iterate the block's outputs, not the symbols it happened to
            # return: an output that does not move is a genuine zero and still
            # has to be recorded, or a later block reading it looks unresolved.
            for o in block_outputs:
                acc = {i: zeros for i in inputs}
                for m, s_om in S.get(o, {}).items():
                    if m not in total:
                        continue                    # a parameter, held fixed
                    for i, t_mi in total[m].items():
                        acc[i] = acc[i] + s_om * t_mi
                total[o] = acc

        return {o: total.get(o, {i: zeros for i in inputs}) for o in outputs}

    def _jacobian(self, ss, inputs, outputs, T, Js, options):
        Js = self._partial_jacobians(ss, inputs, outputs, T, Js, options)

        original_outputs = outputs
        # Name it after this block, as leaf blocks do, so that a precomputed
        # Jacobian carries the identity of what produced it into any DAG it
        # is later embedded in.
        total_Js = JacobianDict.identity(inputs, name=self.name)

        # TODO: horrible, redoing work from partial_jacobians, also need more efficient sifting of intermediates!
        vector_valued = ss._vector_valued()
        inputs = (inputs | self._required) - vector_valued
        outputs = (outputs | self._required) - vector_valued
        for block in self.blocks:
            if (inputs & block.inputs) and (outputs & block.outputs):
                J = block.jacobian(ss, inputs & block.inputs, outputs & block.outputs, T, Js, options)
                total_Js.update(J @ total_Js)

        return total_Js[original_outputs & total_Js.outputs, :]

# Useful type aliases
Model = CombinedBlock
