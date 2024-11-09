from typing import List, Optional, Dict
import numpy as np
import copy

from sequence_jacobian.blocks.support.law_of_motion import DiscreteChoice
from ...utilities.function import ExtendedFunction, CombinedExtendedFunction
from ...utilities.ordered_set import OrderedSet
from ...utilities.misc import make_tuple, logit_choice
from .law_of_motion import (lottery_1d, ShockedPolicyLottery1D,
                            lottery_2d, ShockedPolicyLottery2D,
                            Markov)


class Stage:
    def backward_step(self, inputs: OrderedSet, return_lom: bool, return_out: bool):
        pass

    def __init__(self, hetoutputs=None):

        # prior to any hetinputs / hetoutputs
        self.original_inputs = self.inputs.copy()
        self.original_outputs = self.outputs.copy()

        if hetoutputs is not None:
            hetoutputs = CombinedExtendedFunction(hetoutputs)
        self.process_hetoutputs(hetoutputs, tocopy=False)

    def process_hetoutputs(self, hetoutputs: Optional[CombinedExtendedFunction], tocopy=True):
        if tocopy:
            self = copy.copy(self)
        self.inputs = self.original_inputs.copy()
        self.outputs = self.original_outputs.copy()

        if hetoutputs is not None:
            self.inputs |= (hetoutputs.inputs - self.outputs)
            self.outputs |= hetoutputs.outputs

        self.hetoutputs = hetoutputs

        return self
    
    def add_hetoutputs(self, functions):
        if self.hetoutputs is None:
            return self.process_hetoutputs(CombinedExtendedFunction(functions))
        else:
            return self.process_hetoutputs(self.hetoutputs.add(functions))

    def remove_hetoutputs(self, names):
        return self.process_hetoutputs(self.hetoutputs.remove(names))
    

class Continuous1D(Stage):
    """Endogeous continuous state variable."""
    def __init__(self, backward, policy, f, name=None, hetoutputs=None, monotonic=False):
        # subclass-specific attributes
        self.f = ExtendedFunction(f)
        self.policy = policy
        self.monotonic = monotonic

        # attributes needed for any stage
        if name is None:
            name = self.f.name
        self.name = name
        self.backward = OrderedSet(make_tuple(backward))
        self.outputs = self.f.outputs - self.backward
        self.inputs = self.f.inputs                      # includes backward

        # process hetoutputs, if any
        super().__init__(hetoutputs)

    def __repr__(self):
        return f"<Stage-Continuous1D '{self.name}' with policy '{self.policy}'>"

    def backward_step(self, inputs, return_lom=False, return_out=False):
        # get v_j
        outputs = self.f(inputs)

        # get Λ_j'
        if return_lom:
            lom = lottery_1d(outputs[self.policy], inputs[self.policy + '_grid'], monotonic=self.monotonic)

        # get y_j
        if return_out and self.hetoutputs is not None:
            outputs.update(self.hetoutputs({**inputs, **outputs}, self.hetoutputs.outputs))
        
        if not return_lom:
            return outputs
        else:
            return outputs, lom
    

class Continuous2D(Stage):
    """Two endogeous continuous state variables."""
    def __init__(self, backward, policy, f, name=None, hetoutputs=None):
        # subclass-specific attributes
        self.f = ExtendedFunction(f)
        self.policy = OrderedSet(policy)

        # attributes needed for any stage
        if name is None:
            name = self.f.name
        self.name = name
        self.backward = OrderedSet(make_tuple(backward))
        self.outputs = self.f.outputs - self.backward
        self.inputs = self.f.inputs

        # process hetoutputs, if any
        super().__init__(hetoutputs)

    def __repr__(self):
        return f"<Stage-Continuous2D '{self.name}' with policies {self.policy}>"

    def backward_step(self, inputs, return_lom=False, return_out=False):
        # get v_j
        outputs = self.f(inputs)

        # get Λ_j'
        if return_lom:
            lom = lottery_2d(outputs[self.policy[0]], outputs[self.policy[1]],
                             inputs[self.policy[0] + '_grid'], inputs[self.policy[1] + '_grid'])

        # get y_j
        if return_out and self.hetoutputs is not None:
            outputs.update(self.hetoutputs({**inputs, **outputs}, self.hetoutputs.outputs))
        
        if not return_lom:
            return outputs
        else:
            return outputs, lom
    

class Exogenous(Stage):
    """Exogenous discrete state variable."""
    def __init__(self, markov_name, index, name, backward, hetoutputs=None):
        # subclass-specific attributes
        self.markov_name = markov_name
        self.index = index

        # attributes needed for any stage
        self.name = name
        self.backward = OrderedSet(make_tuple(backward))
        self.outputs = OrderedSet([])
        self.inputs = self.backward | [markov_name]

        # process hetoutputs, if any
        super().__init__(hetoutputs)

    def __repr__(self):
        return f"<Stage-Exogenous '{self.name}' with Markov matrix '{self.markov_name}'>"
    
    def backward_step(self, inputs, return_lom=False, return_out=False):
        # get Λ_j'
        if len(inputs[self.markov_name].shape) == 2:
            # transpose needed because we usually declare Pi(z, z') in backward mode
            Pi = Markov(self.index, inputs[self.markov_name].T)         # Pi(z', z)
        elif len(inputs[self.markov_name].shape) > 2:
            Pi = DiscreteChoice(self.index, inputs[self.markov_name])   # Pi(z'|e, z, a)

        # get v_j
        outputs = {k: Pi.T @ inputs[k] for k in self.backward}

        # get y_j
        if return_out and self.hetoutputs is not None:
            outputs.update(self.hetoutputs({**inputs, **outputs}, self.hetoutputs.outputs))

        if not return_lom:
            return outputs
        else:
            return outputs, Pi
        

class LogitChoice(Stage):
    """Endogenous discrete choice with type 1 extreme value taste shocks."""
    def __init__(self, value, backward, index, taste_shock_scale, f=None, name=None, hetoutputs=None):
        # flow utility
        if f is not None:
            f = ExtendedFunction(f)
            if not len(f.outputs) == 1:
                raise ValueError(f'Flow utility function {f.name} returning multiple outputs {f.outputs}')
            self.f = f
        else:
            self.f = None

        # other subclass-specific attributes
        self.index = index
        self.value = value
        self.backward = OrderedSet(make_tuple(backward)) | [value]
        self.taste_shock_scale = taste_shock_scale

        # attributes needed for any stage
        if name is None:
            name = self.f.name
        self.name = name
        self.outputs = OrderedSet([])
        self.inputs = self.backward | [value, taste_shock_scale]
        if f is not None:
            self.inputs |= f.inputs

        # process hetoutputs, if any
        super().__init__(hetoutputs)

    def __repr__(self):
        return f"<Stage-Discrete '{self.name}'>"

    def backward_step(self, inputs, return_lom=False, return_out=False):
        # initialize discrete choice-specific continuation value
        V_next = inputs[self.value]                     # V'(z, e', a)
        V_next = V_next[np.newaxis, ...]                # V'(e, z, e', a)
        V_next = np.swapaxes(V_next, 0, self.index+1)   # V'(e', z, e, a)

        # add flow utility
        if self.f is not None:
            flow_u = self.f(inputs)                     # u(e', z, e, a)
            flow_u = next(iter(flow_u.values()))
            V_next = flow_u + V_next                    # u(e', z, e, a) + V'(e', z, e, a)
        
        # get v_j(value function), Λ_j' jointly
        P, V = logit_choice(V_next, inputs[self.taste_shock_scale])
        lom = DiscreteChoice(self.index, P)
        
        # get v_j(non value function)
        outputs = {k: lom.T @ inputs[k] for k in self.backward - [self.value]}
        outputs[self.value] = V
        outputs['P'] = P                                # save P(e'|z, e, a) 

        # get y_j
        if return_out and self.hetoutputs is not None:
            outputs.update(self.hetoutputs({**inputs, **outputs}, self.hetoutputs.outputs))

        if not return_lom:
            return outputs
        else:
            return outputs, lom