"""Primitives to provide clarity and structure on blocks/models work"""

import warnings

import numpy as np
from numbers import Real
from typing import Any, Dict, Union, Tuple, Optional, List
from copy import deepcopy

from .support.steady_state import provide_solver_default, solve_for_unknowns, compute_target_values
from .support.parent import Parent
from ..utilities import misc
from ..utilities.function import input_defaults
from ..utilities.bijection import Bijection
from ..utilities.ordered_set import OrderedSet
from ..classes import SteadyStateDict, UserProvidedSS, ImpulseDict, JacobianDict, FactoredJacobianDict
from ..determinacy import winding_number_of_path

Array = Any

class Block:
    """The abstract base class for all `Block` objects."""

    def __init__(self):
        self.M = Bijection({})

        self.steady_state_options = self.input_defaults_smart('_steady_state')
        self.impulse_nonlinear_options = self.input_defaults_smart('_impulse_nonlinear')
        self.impulse_linear_options = self.input_defaults_smart('_impulse_linear')
        self.jacobian_options = self.input_defaults_smart('_jacobian')
        self.partial_jacobians_options = self.input_defaults_smart('_partial_jacobians')
    
    def inputs(self):
        pass

    def outputs(self):
        pass

    def steady_state(self, calibration: Union[SteadyStateDict, UserProvidedSS], 
                     dissolve: List[str] = [], options: Dict[str, dict] = {}, **kwargs) -> SteadyStateDict:
        """Evaluate a partial equilibrium steady state of Block given a `calibration`."""
        inputs = self.inputs.copy()
        if isinstance(self, Parent):
            for k in dissolve:
                inputs |= self.get_attribute(k, 'unknowns').keys()

        calibration = SteadyStateDict(calibration)[inputs]
        own_options = self.get_options(options, kwargs, 'steady_state')
        if isinstance(self, Parent):
            return self.M @ self._steady_state(self.M.inv @ calibration, dissolve=dissolve,
                                               options=options, **own_options)
        else:
            return self.M @ self._steady_state(self.M.inv @ calibration, **own_options)

    def impulse_nonlinear(self, ss: SteadyStateDict, inputs: Union[Dict[str, Array], ImpulseDict],
                          outputs: Optional[List[str]] = None,
                          internals: Union[Dict[str, List[str]], List[str]] = {},
                          Js: Dict[str, JacobianDict] = {}, options: Dict[str, dict] = {},
                          ss_initial: Optional[SteadyStateDict] = None, **kwargs) -> ImpulseDict:
        """Calculate a partial equilibrium, non-linear impulse response of `outputs` to a set of shocks in `inputs`
        around a steady state `ss`."""
        own_options = self.get_options(options, kwargs, 'impulse_nonlinear')
        inputs = ImpulseDict(inputs)
        actual_outputs, inputs_as_outputs = self.process_outputs(ss,
            self.make_ordered_set(inputs), self.make_ordered_set(outputs))
        
        if isinstance(self, Parent):
            # SolvedBlocks may use Js and may be nested in a CombinedBlock, so we need to pass them down to any parent
            out = self.M @ self._impulse_nonlinear(self.M.inv @ ss, self.M.inv @ inputs, self.M.inv @ actual_outputs, internals, Js, options, self.M.inv @ ss_initial, **own_options)
        elif hasattr(self, 'internals'):
            out = self.M @ self._impulse_nonlinear(self.M.inv @ ss, self.M.inv @ inputs, self.M.inv @ actual_outputs, self.internals_to_report(internals), self.M.inv @ ss_initial, **own_options)
        else:
            out = self.M @ self._impulse_nonlinear(self.M.inv @ ss, self.M.inv @ inputs, self.M.inv @ actual_outputs, self.M.inv @ ss_initial, **own_options)

        return inputs[inputs_as_outputs] | out

    def impulse_linear(self, ss: SteadyStateDict, inputs: Union[Dict[str, Array], ImpulseDict],
                       outputs: Optional[List[str]] = None, Js: Dict[str, JacobianDict] = {},
                       options: Dict[str, dict] = {}, **kwargs) -> ImpulseDict:
        """Calculate a partial equilibrium, linear impulse response of `outputs` to a set of shocks in `inputs`
        around a steady state `ss`."""
        own_options = self.get_options(options, kwargs, 'impulse_linear')
        inputs = ImpulseDict(inputs)
        actual_outputs, inputs_as_outputs = self.process_outputs(ss, self.make_ordered_set(inputs), self.make_ordered_set(outputs))

        if isinstance(self, Parent):
            out = self.M @ self._impulse_linear(self.M.inv @ ss, self.M.inv @ inputs, self.M.inv @ actual_outputs, Js, options, **own_options)
        else:
            out = self.M @ self._impulse_linear(self.M.inv @ ss, self.M.inv @ inputs, self.M.inv @ actual_outputs, Js, **own_options)

        return inputs[inputs_as_outputs] | out

    def partial_jacobians(self, ss: SteadyStateDict, inputs: Optional[List[str]] = None, outputs: Optional[List[str]] = None,
                          T: Optional[int] = None, Js: Dict[str, JacobianDict] = {}, options: Dict[str, dict] = {}, **kwargs):
        if inputs is None:
            inputs = self.inputs
        if outputs is None:
            outputs = self.outputs
        
        # if you have a J for this block that already has everything you need, use it
        # TODO: add check for T, maybe look at verify_saved_jacobian for ideas?
        if (self.name in Js) and isinstance(Js[self.name], JacobianDict) and (inputs <= Js[self.name].inputs) and (outputs <= Js[self.name].outputs):
            return {self.name: Js[self.name][outputs, inputs]}

        # if it's a leaf, just call Jacobian method, include if nonzero
        if not isinstance(self, Parent):
            own_options = self.get_options(options, kwargs, 'jacobian')
            jac = self.jacobian(ss, inputs, outputs, T, **own_options)
            return {self.name: jac} if jac else {}

        # otherwise call child method with remapping (and remap your own but none of the child Js)
        own_options = self.get_options(options, kwargs, 'partial_jacobians')
        partial = self._partial_jacobians(self.M.inv @ ss, self.M.inv @ inputs, self.M.inv @ outputs, T, Js, options, **own_options)
        if self.name in partial:
            partial[self.name] = self.M @ partial[self.name]
        return partial

    def jacobian(self, ss: SteadyStateDict, inputs: List[str],
                 outputs: Optional[List[str]] = None,
                 T: Optional[int] = None, Js: Dict[str, JacobianDict] = {},
                 options: Dict[str, dict] = {}, **kwargs) -> JacobianDict:
        """Calculate a partial equilibrium Jacobian to a set of `input` shocks at a steady state `ss`."""
        own_options = self.get_options(options, kwargs, 'jacobian')
        inputs = self.make_ordered_set(inputs)
        outputs, _ = self.process_outputs(ss, {}, self.make_ordered_set(outputs))

        # if you have a J for this block that has everything you need, use it
        if (self.name in Js) and isinstance(Js[self.name], JacobianDict) and (inputs <= Js[self.name].inputs) and (outputs <= Js[self.name].outputs):
            return Js[self.name][outputs, inputs]
        
        # if it's a leaf, call Jacobian method, don't supply Js
        if not isinstance(self, Parent):
            return self.M @ self._jacobian(self.M.inv @ ss, self.M.inv @ inputs, self.M.inv @ outputs, T, **own_options)
        
        # otherwise remap own J (currently needed for SolvedBlock only)
        Js = Js.copy()
        if self.name in Js:
            Js[self.name] = self.M.inv @ Js[self.name]
        return self.M @ self._jacobian(self.M.inv @ ss, self.M.inv @ inputs, self.M.inv @ outputs, T=T, Js=Js, options=options, **own_options)

    solve_steady_state_options = dict(solver="", solver_kwargs={}, ttol=1e-12, ctol=1e-9,
        verbose=False, constrained_method="linear_continuation", constrained_kwargs={})

    def solve_steady_state(self, calibration: Dict[str, Union[Real, Array]],
                           unknowns: Dict[str, Union[Real, Tuple[Real, Real]]],
                           targets: Union[Array, Dict[str, Union[str, Real]]],
                           dissolve: List = [], options: Dict[str, dict] = {}, **kwargs):
        """Evaluate a general equilibrium steady state of Block given a `calibration`
        and a set of `unknowns` and `targets` corresponding to the endogenous variables to be solved for and
        the target conditions that must hold in general equilibrium"""
        opts = self.get_options(options, kwargs, 'solve_steady_state')

        ss =  SteadyStateDict(calibration)

        solver = opts['solver'] if opts['solver'] else provide_solver_default(unknowns)

        def residual(unknown_values, unknowns_keys=unknowns.keys(), targets=targets):
            ss.update(misc.smart_zip(unknowns_keys, unknown_values))
            ss.update(self.steady_state(ss, dissolve=dissolve, options=options, **kwargs))
            return compute_target_values(targets, ss)

        _ = solve_for_unknowns(residual, unknowns, solver, opts['solver_kwargs'],
                               tol=opts['ttol'], verbose=opts['verbose'],
                               constrained_method=opts['constrained_method'],
                               constrained_kwargs=opts['constrained_kwargs'])

        return ss

    solve_impulse_nonlinear_options = dict(tol=1E-8, maxit=30, verbose=True)

    def solve_impulse_nonlinear(self, ss: SteadyStateDict, unknowns: List[str], targets: List[str],
                                inputs: Union[Dict[str, Array], ImpulseDict], outputs: Optional[List[str]] = None,
                                internals: Union[Dict[str, List[str]], List[str]] = {}, Js: Dict[str, JacobianDict] = {}, 
                                options: Dict[str, dict] = {}, H_U_factored: Optional[FactoredJacobianDict] = None,
                                ss_initial: Optional[SteadyStateDict] = None, **kwargs) -> ImpulseDict:
        """Calculate a general equilibrium, non-linear impulse response to a set of shocks in `inputs` 
           around a steady state `ss`, given a set of `unknowns` and `targets` corresponding to the endogenous
           variables to be solved for and the `targets` that must hold in general equilibrium"""
        inputs = ImpulseDict(inputs)
        unknowns, targets = OrderedSet(unknowns), OrderedSet(targets)

        input_names = self.make_ordered_set(inputs)
        actual_outputs, inputs_as_outputs = self.process_outputs(ss, input_names | unknowns, self.make_ordered_set(outputs))

        T = inputs.T

        Js = self.partial_jacobians(ss, input_names | unknowns, (actual_outputs | targets) - unknowns, T, Js, options, **kwargs)

        if H_U_factored is None:
            H_U = self.jacobian(ss, unknowns, targets, T, Js, options, **kwargs)
            H_U_factored = FactoredJacobianDict(H_U, T)

        opts = self.get_options(options, kwargs, 'solve_impulse_nonlinear')

        # Newton's method
        U = ImpulseDict({k: np.zeros(T) for k in unknowns})
        if opts['verbose']:
            print(f'Solving {self.name} for {unknowns} to hit {targets}')
        for it in range(opts['maxit']):
            results = self.impulse_nonlinear(ss, inputs | U, actual_outputs | targets, internals, Js, options, ss_initial, **kwargs)
            errors = {k: np.max(np.abs(results[k])) for k in targets}
            if opts['verbose']:
                print(f'On iteration {it}')
                for k in errors:
                    print(f'   max error for {k} is {errors[k]:.2E}')
            if all(v < opts['tol'] for v in errors.values()):
                break
            else:
                U += H_U_factored.apply(results)
        else:
            raise ValueError(f'No convergence after {opts["maxit"]} backward iterations!')

        return (inputs | U)[inputs_as_outputs] | results

    solve_impulse_linear_options = {}

    def solve_impulse_linear(self, ss: SteadyStateDict, unknowns: List[str], targets: List[str],
                             inputs: Union[Dict[str, Array], ImpulseDict], outputs: Optional[List[str]] = None,
                             Js: Optional[Dict[str, JacobianDict]] = {}, options: Dict[str, dict] = {},
                             H_U_factored: Optional[FactoredJacobianDict] = None, **kwargs) -> ImpulseDict:

        """Calculate a general equilibrium, linear impulse response to a set of shocks in `inputs`
           around a steady state `ss`, given a set of `unknowns` and `targets` corresponding to the endogenous
           variables to be solved for and the target conditions that must hold in general equilibrium"""
        inputs = ImpulseDict(inputs)
        unknowns, targets = OrderedSet(unknowns), OrderedSet(targets)

        input_names = self.make_ordered_set(inputs)
        actual_outputs, inputs_as_outputs = self.process_outputs(ss, input_names | unknowns, self.make_ordered_set(outputs))

        T = inputs.T

        Js = self.partial_jacobians(ss, input_names | unknowns, (actual_outputs | targets) - unknowns, T, Js, options, **kwargs)

        dH = self.impulse_linear(ss, inputs, targets, Js, options, **kwargs).get(targets) # .get(targets) fills in zeros

        if H_U_factored is None:
            H_U = self.jacobian(ss, unknowns, targets, T, Js, options, **kwargs).pack(T)
            dU = ImpulseDict.unpack(-np.linalg.solve(H_U, dH.pack()), unknowns, T)
        else:
            dU = H_U_factored @ dH

        return (inputs | dU)[inputs_as_outputs] | self.impulse_linear(ss, dU | inputs, actual_outputs, Js, options, **kwargs)

    solve_jacobian_options = {}

    def solve_jacobian(self, ss: SteadyStateDict, unknowns: List[str], targets: List[str],
                       inputs: List[str], outputs: Optional[List[str]] = None, T: int = 300,
                       Js: Dict[str, JacobianDict] = {}, options: Dict[str, dict] = {},
                       H_U_factored: Optional[FactoredJacobianDict] = None, **kwargs) -> JacobianDict:
        """Calculate a general equilibrium Jacobian to a set of `exogenous` shocks
        at a steady state `ss`, given a set of `unknowns` and `targets` corresponding to the endogenous
        variables to be solved for and the target conditions that must hold in general equilibrium"""
        inputs, unknowns = self.make_ordered_set(inputs), self.make_ordered_set(unknowns)
        actual_outputs, unknowns_as_outputs = self.process_outputs(ss, unknowns, self.make_ordered_set(outputs))

        Js = self.partial_jacobians(ss, inputs | unknowns, (actual_outputs | targets) - unknowns, T, Js, options, **kwargs)
        
        H_Z = self.jacobian(ss, inputs, targets, T, Js, options, **kwargs)

        if H_U_factored is None:
            H_U = self.jacobian(ss, unknowns, targets, T, Js, options, **kwargs).pack(T)
            U_Z = JacobianDict.unpack(-np.linalg.solve(H_U, H_Z.pack(T)), unknowns, inputs, T)
        else:
            U_Z = H_U_factored @ H_Z
        # Named for both branches: U_Z is combined into a DAG just below, where an
        # anonymous block would collide with any other anonymous one.
        U_Z.name = f'{self.name}_U_Z'

        from sequence_jacobian import combine
        self_with_unknowns = combine([U_Z, self])
        return self_with_unknowns.jacobian(ss, inputs, unknowns_as_outputs | actual_outputs, T, Js, options, **kwargs)

    def _promote_solved(self):
        """Primitive blocks, with solved unknowns and targets lifted out.

        A solved block solves internally, which is an operator inverse the
        symbol cannot see through, and imposes its own boundary condition --
        precisely what could hide an indeterminacy. Unwrapping it and promoting
        its unknowns to the model's describes the same equilibrium, as the same
        equations solved jointly rather than nested.

        Returns the blocks, and the unknowns and targets to append.
        """
        from .combined_block import CombinedBlock
        from .solved_block import SolvedBlock

        blocks, unknowns, targets = [], [], []

        def walk(block):
            if isinstance(block, SolvedBlock):
                unknowns.extend(block.unknowns)
                targets.extend(block.targets)
                walk(block.block)
            elif isinstance(block, CombinedBlock):
                for kid in block.blocks:
                    walk(kid)
            else:
                blocks.append(block)

        walk(self)
        return blocks, unknowns, targets

    def winding_number(self, ss: SteadyStateDict, unknowns: List[str],
                       targets: List[str], T: Optional[int] = None,
                       n_sample: int = 8192, return_path: bool = False):
        """Whether this model has a unique bounded solution, by winding number.

        Returns a WindingNumber -- an int that explains its own reading; see
        that class for what the value means and for the one way zero can
        mislead. With return_path, also returns det j(z) sampled counterclockwise
        around the unit circle, which is worth plotting: how close it passes to
        the origin says how much to trust the count.

        The criterion assumes det j(z) has no zero on the unit circle. What
        that costs, when violated, is the count itself: the argument jumps by pi
        as the path passes through the origin, and no sampling resolves it. So
        the guard is on how far the argument turns between neighbouring samples,
        not on how close the path comes to the origin. The latter turns out not
        to discriminate -- correctly classified models routinely pass within
        1e-3 of the origin relative to their own scale, no further away than
        genuinely broken ones -- whereas the angular step separates the two
        cleanly. Use return_path if you want to judge proximity yourself.

        Solved blocks are unwrapped and their unknowns and targets promoted
        automatically, so pass only the model's own. That is not a convenience:
        the problem a solved block solves internally has a winding number of its
        own, and it belongs in the count. Giving the solved block a symbol by the
        implicit function theorem instead, and winding the outer system alone,
        would divide det j(z) by that of the inner problem, and so leave the
        count off by exactly the inner winding number -- harmless when the solved
        block is determinate on its own, wrong in either direction when it is
        not, and without any warning. test_promotion_matches_writing_the_model_flat
        has a two-equation example.
        """
        from .combined_block import combine

        blocks, extra_unknowns, extra_targets = self._promote_solved()
        unknowns = self.make_ordered_set(list(unknowns) + extra_unknowns)
        targets = self.make_ordered_set(list(targets) + extra_targets)

        if len(unknowns) != len(targets):
            raise ValueError(f'H_U is not square: {len(unknowns)} unknowns '
                             f'{list(unknowns)} against {len(targets)} targets '
                             f'{list(targets)}')

        flat = combine(blocks, name=f'{self.name}_promoted')
        symbols = flat._symbol(ss, unknowns, targets, T, n_sample)

        H_U = np.empty((n_sample, len(targets), len(unknowns)), dtype=complex)
        for a, o in enumerate(targets):
            for b, i in enumerate(unknowns):
                H_U[:, a, b] = symbols[o][i]

        path = np.linalg.det(H_U)

        if np.abs(path).max() == 0.0:
            raise ValueError(
                'det j(z) vanishes identically, so H_U is singular at every z '
                'and has no winding number. Usually a target that does not '
                'depend on any unknown, or two unknowns entering identically.')

        wind = winding_number_of_path(path)
        return (wind, path) if return_path else wind

    def check_existence(self, ss: SteadyStateDict, unknowns: List[str],
                        targets: List[str], T: Optional[int] = None,
                        T_compare: Optional[int] = None, n_draws: int = 100,
                        decay: Tuple[float, float] = (0.85, 0.95),
                        seed: Optional[int] = 0, growth_tol: float = 10.0):
        """Probe whether H_U x = y actually has solutions, for the false positives.

        A winding number of zero is only *generically* sufficient. The known
        economic counterexample is the intertemporal Keynesian cross in goods
        space, where the present-value budget constraint leaves the operator a
        left null vector, so its range holds only sequences with zero net
        present value -- not invertible, yet winding number zero. Since
        non-genericity always produces nonexistence and indeterminacy together,
        ruling out nonexistence rules out both.

        The probe is from Auclert, Majic, Rognlie and Straub (2025), section 3.5.
        Feed in random decaying sequences and solve the truncated system. If no
        true solution exists, the truncated solve returns something enormous
        relative to what went in, and -- the actual signature -- it gets worse
        as T grows, because nothing about it is converging. A genuine solution
        barely moves.

        Returns the growth factor: how much the solution magnitude grows when T
        goes from T to T_compare. Around 1 means existence looks fine. Orders of
        magnitude means it does not; the paper's example goes from roughly 2e3 to
        1e9 on doubling.

        Costs a dense solve of size (unknowns x T_compare) squared, so doubling T
        on a large model is expensive; set T_compare yourself to control it.

        The model has to be buildable at both horizons. A precomputed Jacobian
        embedded in the DAG carries its own fixed T and cannot be rebuilt at
        another, so this raises rather than comparing two different models.
        """
        from .combined_block import combine

        blocks, extra_unknowns, extra_targets = self._promote_solved()
        unknowns = self.make_ordered_set(list(unknowns) + extra_unknowns)
        targets = self.make_ordered_set(list(targets) + extra_targets)
        flat = combine(blocks, name=f'{self.name}_promoted')

        if T_compare is None:
            T_compare = 2 * T

        rng = np.random.default_rng(seed)
        n = len(unknowns)
        lo, hi = decay
        eps = rng.uniform(lo, hi, size=(2, n, n_draws))

        def magnitude(horizon):
            t = np.arange(horizon)[:, None, None]
            # one decaying sequence per target per draw, stacked into y
            y = (eps[0] ** t - 0.5 * eps[1] ** t)         # (horizon, n, n_draws)
            y = y.transpose(1, 0, 2).reshape(n * horizon, n_draws)

            try:
                H_U = flat.jacobian(ss, unknowns, targets, T=horizon).pack(horizon)
            except ValueError as e:
                raise ValueError(
                    f'Cannot rebuild this model at T={horizon}, which the probe '
                    'needs in order to watch how the solution scales with the '
                    'horizon. A precomputed Jacobian embedded in the DAG has its '
                    'own fixed T, so pass T and T_compare that match it, or build '
                    'the model from blocks that can be evaluated at any horizon.'
                ) from e
            try:
                x = np.linalg.solve(H_U, y)
            except np.linalg.LinAlgError:
                # Exactly singular rather than merely blowing up. The lag
                # operator does this: its truncation has an empty first row.
                return np.inf
            return np.max(np.abs(x)) / np.max(np.abs(y))

        small, large = magnitude(T), magnitude(T_compare)
        if not np.isfinite(small) or not np.isfinite(large):
            warnings.warn(
                'The truncated system is exactly singular, so H_U x = y has no '
                'solution for generic y.', stacklevel=2)
            return np.inf
        growth = large / small

        if growth > growth_tol:
            warnings.warn(
                f'The truncated solution grows by a factor of {growth:.1e} when T '
                f'goes from {T} to {T_compare}, reaching {large:.1e} times the '
                'size of what was fed in. That is the signature of no bounded '
                'solution existing, so a winding number of zero here would be a '
                'false positive -- see check_existence.', stacklevel=2)

        return growth

    def solved(self, unknowns, targets, name=None, solver=None, solver_kwargs=None):
        if name is None:
            name = self.name + "_solved"
        from .solved_block import SolvedBlock
        return SolvedBlock(self, name, unknowns, targets, solver, solver_kwargs)

    def remap(self, map: Dict[str, str]):
        other = deepcopy(self)
        other.M = self.M @ Bijection(map)
        other.inputs = other.M @ self.inputs
        other.outputs = other.M @ self.outputs
        if hasattr(self, 'input_list'):
            other.input_list = other.M @ self.input_list
        if hasattr(self, 'output_list'):
            other.output_list = other.M @ self.output_list
        if hasattr(self, 'non_back_iter_outputs'):
            other.non_back_iter_outputs = other.M @ self.non_back_iter_outputs
        return other

    def rename(self, name: Optional[str] = None, suffix: Optional[str] = None):
        """Convention: specify suffix kwarg if called on Parent."""
        if isinstance(self, Parent):
            other = deepcopy(self)
            other.name = self.name + suffix
            if hasattr(self, 'blocks'):
                other.blocks = [b.rename(name, suffix) for b in self.blocks]
                Parent.__init__(other, other.blocks)
            elif hasattr(self, 'block'):
                other.block = self.block.rename_top(self.block.name + suffix)
                Parent.__init__(other, [other.block])
            return other
        else:
            if suffix is None:
                # called rename on singleton block 
                return self.rename_top(name)  
            else:
                # called rename on Parent, reached leaf
                return self.rename_top(self.name + suffix)

    def rename_top(self, name: str):
        other = deepcopy(self)
        other.name = name
        return other

    def default_inputs_outputs(self, ss: SteadyStateDict, inputs, outputs):
        # TODO: there should be checks to make sure you don't ask for multidimensional stuff for Jacobians?
        # should you be allowed to ask for it (even if not default) for impulses?
        if inputs is None:
            inputs = self.inputs
        if outputs is None:
            outputs = self.outputs - ss._vector_valued()
        return OrderedSet(inputs), OrderedSet(outputs)

    def process_outputs(self, ss, inputs: OrderedSet, outputs: Optional[OrderedSet]):
        if outputs is None:
            actual_outputs = self.outputs - ss._vector_valued()
            inputs_as_outputs = inputs
        else:
            actual_outputs = outputs & self.outputs
            inputs_as_outputs = outputs & inputs
        
        return actual_outputs, inputs_as_outputs

    @staticmethod
    def make_ordered_set(x):
        if x is not None and not isinstance(x, OrderedSet):
            return OrderedSet(x)
        else:
            return x

    def get_options(self, options: dict, kwargs, method):
        own_options = getattr(self, method + "_options")

        if self.name in options:
            merged = {**own_options, **options[self.name], **kwargs}
        else:
            merged = {**own_options, **kwargs}
        
        return {k: merged[k] for k in own_options}

    def input_defaults_smart(self, methodname):
        method = getattr(self, methodname, None)
        if method is None:
            return {}
        else:
            return input_defaults(method)

    def internals_to_report(self, internals):
        if self.name in internals:
            if isinstance(internals, dict):
                # if internals is a dict, we've specified which internals we want from each block
                return internals[self.name]
            else:
                # otherwise internals is some kind of iterable or set, and if we're in it, we want everything
                return self.internals
        else:
            return []
