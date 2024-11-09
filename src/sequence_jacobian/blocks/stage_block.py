from typing import List, Optional
import numpy as np
import copy

from .block import Block
from .het_block import HetBlock
from ..classes import SteadyStateDict, JacobianDict, ImpulseDict
from ..utilities.ordered_set import OrderedSet
from ..utilities.function import ExtendedFunction, CombinedExtendedFunction
from ..utilities.bijection import Bijection
from ..utilities.optimized_routines import within_tolerance
from .. import utilities as utils
from .support.law_of_motion import LawOfMotion
from .support.stages import Stage


class StageBlock(Block):
    def __init__(self, stages: List[Stage], backward_init=None, hetinputs=None, name=None):
        super().__init__()
        inputs = OrderedSet([])   # micro & macro inputs except backward
        outputs = OrderedSet([])  # macro outputs

        for stage in stages:
            # all stages should have the same backward variables
            inputs |= (stage.inputs - stages[-1].backward)
            outputs |= stage.outputs
        
        self.constructor_checks(stages, inputs, outputs)
        self.stages = stages
        self.inputs = inputs
        self.outputs = OrderedSet([o.upper() for o in outputs])
        self.internals = OrderedSet(['law_of_motion']) | [s.name for s in self.stages]
        
        # store "original" copies of these for use whenever we process new hetinputs
        self.original_inputs = self.inputs
        self.original_outputs = self.outputs
        self.original_internals = self.internals

        if name is None:
            name = stages[0].name + "_to_" + stages[-1].name
        self.name = name

        if hetinputs is not None:
            hetinputs = CombinedExtendedFunction(hetinputs)
        self.process_hetinputs(hetinputs, tocopy=False)

        if backward_init is not None:
            backward_init = ExtendedFunction(backward_init)
        self.backward_init = backward_init

    @staticmethod
    def constructor_checks(stages, inputs, outputs):
        backward_all = set().union(*(stage.backward for stage in stages))
        # inputs, outputs, and combined backward should not overlap at all
        if not inputs.isdisjoint(outputs):
            raise ValueError(f'inputs and outputs have overlap {inputs & outputs}')
        if not inputs.isdisjoint(backward_all):
            raise ValueError(f'Inputs and backward have overlap {inputs & backward_all}')
        if not outputs.isdisjoint(backward_all):
            raise ValueError(f'Outputs and backward have overlap {outputs & backward_all}')
        
        # all stages should have the same backward
        # 'p', 'D', 'law_of_motion' are protected names; outputs should not be upper case
        for stage in stages:
            if (backward_all - stage.backward) != OrderedSet({}):
                raise ValueError(f"Stage '{stage.name}' does not have all backward")
            if stage.name in ['P', 'D', 'law_of_motion']:
                raise ValueError(f"Stage '{stage.name}' has invalid name")
            for o in stage.outputs:
                if o in ['p', 'd', 'law_of_motion']:
                    raise ValueError(f"Stages are not allowed to return outputs called 'p', 'd', or 'law_of_motion' but stage '{stage.name}' does")
                if o.isupper(): 
                    raise ValueError(f"Stages are not allowed to report upper-case outputs. Stage '{stage.name}' has an output '{o}'")
                
    def __repr__(self):
        return f"<StageBlock '{self.name}' with stages {[k.name for k in self.stages]}>"
    
    def _steady_state(self, calibration, backward_tol=1E-9, backward_maxit=5000,
                      forward_tol=1E-10, forward_maxit=100_000):
        ss = self.extract_ss_dict(calibration)
        hetinputs = self.return_hetinputs(ss)
        ss.update(hetinputs)
        self.initialize_backward(ss)

        # get v_j, y_j, Λ_j' for j = 0, 1, ..., J-1
        backward, lom = self.backward_steady_state(ss, backward_tol, backward_maxit)
        
        # get D_j for j = 0, 1, ..., J-1
        D = self.forward_steady_state(ss, lom, forward_tol, forward_maxit)
        
        aggregates = {}
        internals = hetinputs   # save everything computed by hetinputs in internals
        for i, stage in enumerate(self.stages):
            # save Y_j
            for k in stage.outputs:
                aggregates[k.upper()] = np.vdot(D[i], backward[i][k])   
            
            # save v_j, y_j, Λ_j', D_j for j = 0, 1, ..., J-1
            internals[stage.name] = {**backward[i], 'law_of_motion': lom[i], 'D': D[i]}

        # save X_j
        for k in self.inputs:
            if np.isscalar(ss[k]):
                aggregates[k] = ss[k]

        return SteadyStateDict(aggregates, {self.name: internals})
    
    def backward_steady_state(self, ss, tol, maxit):
        # initial guess of v_0
        backward = {k: ss[k] for k in self.stages[0].backward}

        # iterate until v_0 converges
        for it in range(maxit):
            backward_new = self.backward_step_ss(backward, ss)
            if it % 10 == 0 and all(within_tolerance(backward_new[k], backward[k], tol) for k in backward):
                break
            backward = backward_new
        else:
            raise ValueError(f'No convergence after {maxit} backward iterations!')
        
        # get v_j, y_j, Λ_j' for all j
        return self.backward_step_ss_report(backward_new, ss)

    def backward_step_ss(self, backward, inputs):
        """Iterate backward through all stages for a single period, ignoring reported outputs"""
        for stage in reversed(self.stages):
            backward = stage.backward_step({**inputs, **backward}, return_lom=False, return_out=False)
        return backward
    
    def backward_step_ss_report(self, backward, inputs):
        backward_all = [backward]           # start from continuation value v_{J}
        lom_all = []
        for stage in reversed(self.stages):
            backward, lom = stage.backward_step({**inputs, **backward}, return_lom=True, return_out=True)
            backward_all.append(backward)   # append v_j, y_j
            lom_all.append(lom)             # append Λ_j'

        # return v_j, y_j, Λ_j' for j = 0, 1, ..., J-1
        return backward_all[::-1][:-1], lom_all[::-1]
    
    def forward_steady_state(self, ss, lom: List[LawOfMotion], tol, maxit):
        # initial guess for D_0
        try:
            D = ss[self.stages[0].name]['D']
        except KeyError:
            # assume D_0 is uniform (ss has the guess v_J, which is on same grid as D_0)
            backward_example = ss[self.stages[0].backward[0]]
            D = np.full(backward_example.shape, 1 / backward_example.size)

        # iterate until D_0 converges
        for it in range(maxit):
            D_new = self.forward_step_steady_state(D, lom)
            if it % 10 == 0 and within_tolerance(D, D_new, tol):
                break
            D = D_new
        else:
            raise ValueError(f'No convergence after {maxit} forward iterations!')

        # return D_j for j = 0, 1, ..., J-1
        return self.forward_step_ss_report(D, lom)

    def forward_step_steady_state(self, D, loms: List[LawOfMotion]):
        for lom in loms:
            D = lom @ D
        return D
    
    def forward_step_ss_report(self, D, loms: List[LawOfMotion]):
        Ds = [D]
        for i, lom in enumerate(loms):
            Ds.append(lom @ Ds[i])
        # return D_j for j=0, 1, ..., J-1
        return Ds[:-1]

    
    '''HetInput and HetOutput options and processing'''
    
    def process_hetinputs(self, hetinputs: Optional[CombinedExtendedFunction], tocopy=True):
        if tocopy:
            self = copy.copy(self)
        inputs = self.original_inputs.copy()
        internals = self.original_internals.copy()

        if hetinputs is not None:
            inputs |= hetinputs.inputs
            inputs -= hetinputs.outputs
            internals |= hetinputs.outputs

        self.inputs = inputs
        self.internals = internals

        self.hetinputs = hetinputs

        return self
    
    def extract_ss_dict(self, ss):
        """Flatten ss dict and internals for this block (if present) into one dict,
        but keeping each stage within internals as a subdict"""
        if isinstance(ss, SteadyStateDict):
            ssnew = ss.toplevel.copy()
            if self.name in ss.internals:
                ssnew.update(ss.internals[self.name])
            return ssnew
        else:
            return ss.copy()
        
    def return_hetinputs(self, d):
        if self.hetinputs is not None:
            return self.hetinputs(d)
        else:
            return {}

    def initialize_backward(self, ss):
        if not all(k in ss for k in self.stages[-1].backward):
            ss.update(self.backward_init(ss))