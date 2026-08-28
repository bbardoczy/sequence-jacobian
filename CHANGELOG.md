# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **About this fork.** This is a fork of
> [shade-econ/sequence-jacobian](https://github.com/shade-econ/sequence-jacobian),
> maintained for personal use and for public replication files. Version numbers are
> independent of upstream: entries below describe changes relative to the previous
> release of *this* repository, starting from upstream `v1.0.0`.

## [Unreleased]

### Added

- **Determinacy checking**, implementing the winding number criterion of Auclert, Majic,
  Rognlie and Straub (2025), *Determinacy and Large-Scale Solutions in the Sequence Space*.
  `Block.winding_number(ss, unknowns, targets)` returns the winding number of `det j(z)`,
  where `j` is the symbol of `H_U`: zero means the model generically has a unique bounded
  solution, a negative number counts the dimensions of indeterminacy, and a positive number
  the dimensions along which no solution exists. The result is an `int` subclass whose repr
  spells out the verdict, and `return_path=True` additionally returns the sampled path for
  plotting. Symbols are built block by block and composed along the DAG, never read off a
  composed Jacobian, and `SolvedBlock`s are unwrapped so their unknowns join the model's.
- `Block.check_existence(ss, unknowns, targets)`, which settles existence without using the
  winding number, following section 3.5 of the same paper. A winding number of zero is only
  generically sufficient, and the standing counterexample is not exotic -- the intertemporal
  Keynesian cross in goods space, where the present-value budget constraint leaves the
  operator a left null vector. Since non-genericity always produces nonexistence and
  indeterminacy together, ruling out nonexistence rules out both.
- `sequence_jacobian.determinacy`, holding the machinery behind those two methods. Nothing
  from it is exported into the top-level namespace, since neither method needs an import.
- `fake_news_matrix()` on `HetBlock` and `StageBlock`, where `F[o][i][t, s]` is the date-`t`
  response of `o` to news at date 0 of a shock to `i` at date `s`. It was already computed
  and then discarded inside `_jacobian`; exposing it duplicates no work and leaves the
  Jacobian path unchanged.
- `notebooks/determinacy.ipynb` (Tutorial 6), and a Jacobian timing benchmark under
  `tests/performance/`.
- A `[tool.ruff]` configuration. Ruff reads `requires-python` only from `pyproject.toml`,
  and this package declares it in `setup.cfg`, so with no pinned `target-version` ruff
  assumed a current interpreter and proposed 3.10-only annotation syntax that does not
  parse on the 3.9 this package supports.

### Changed

- **`JacobianDictBlock` now requires a `name`.** A block inside a DAG is addressed by name --
  it keys `kids`, `descendants`, and any `Js` supplied to reuse a Jacobian -- so two
  anonymous blocks are indistinguishable. Constructing one without a name now raises, with
  a message naming the fix from the caller's side.
- **Duplicate block names are now rejected.** `Parent` previously deduplicated only among
  `Parent` kids, so two same-named leaf blocks were absorbed silently: `kids` kept whichever
  came last and the other was dropped from addressing, while both still evaluated.
- Jacobians returned by `CombinedBlock` now carry the block's own name rather than falling
  back to `"NestedDict"`, and that name survives slicing and merging. `JacobianDict.identity`
  and `unpack` take an optional `name`. Anonymous Jacobians collided with each other, which
  made `solve_jacobian` unusable on any model built around a precomputed combined-block
  Jacobian, raising `Overlapping block name NestedDict`.

### Fixed

- `SimpleSparse.array()` raised `ValueError` on an operator with no elements, which is what
  `__add__` leaves behind when terms cancel exactly -- a government budget where interest
  paid on debt nets against interest received on the same debt, for instance. This broke
  every product against a dense matrix, so it surfaced whenever such an operator met a
  heterogeneous-agent Jacobian during composition.
- `apply_function` accepted `**kwargs` and silently discarded them whenever any argument
  was a `Displace` -- the only case the function exists for. `np.round(x, decimals=2)`
  applied through it returned values rounded to zero decimals instead, with no error, and
  the same call on a plain array honoured the keyword, so results depended on the argument
  type. This is the same defect Gregor Boehl fixed in `Displace.apply` for 1.1.0; the free
  function was missed. `vectorize_func_over_time` now takes and forwards keyword arguments
  too, and both are covered by tests.
- `tests/utils/test_DAG.py::test_visited` asserted nothing: five bare comparisons whose
  results were discarded, so it had passed unconditionally since it was written. All five
  conditions hold, so `visit_from_outputs` and `visit_from_inputs` are newly covered rather
  than newly correct.

## [1.1.0] - 2026-08-18

### Added

- `ConditionalMarkov`, a new `LawOfMotion` subclass. `ExogenousMaker` now also accepts
  Markov matrices that depend on other states.
- `StageBlock` can report a narrowed set of internals (`law_of_motion` only, or a
  chosen subset of stages) instead of always returning everything.
- `environment.yml` describing a reproducible conda-forge development environment,
  including an editable install of the package itself.

### Changed

- **Arithmetic on `ImpulseDict` no longer applies to `internals`.** Binary operations
  (against another `ImpulseDict`, a `SteadyStateDict`, or a scalar) and unary operations
  now act on top-level series only and pass `internals` through unchanged. In particular,
  `Block.impulse_nonlinear` no longer demeans internals, and `StageBlock` returns all of
  them. This changes returned numbers rather than raising an error, so results computed
  with 1.0.0 that rely on demeaned internals will differ.
- Applying a `Bijection` to a `dict` now prioritizes remapped names when a key clash
  occurs, instead of letting an unmapped key overwrite a remapped one.
- Minimum supported Python is now 3.9 (was 3.7, which had become untestable).
  Dependency floors raised to the oldest releases that actually install on Python 3.9:
  numpy 1.19.3, scipy 1.5.4, numba 0.53.
- CI now tests Python 3.9-3.13 on both Linux and Windows.
- `setup.py` reduced to a thin shim; all packaging metadata now lives in `setup.cfg`,
  which previously was silently overridden.

### Removed

- `requirements.txt`. Runtime dependencies are declared in `setup.cfg` under
  `install_requires`.

### Fixed

- `ss_initial` was not loaded correctly in `StageBlock.impulse_nonlinear`.
- `curlyY` is now initialized to zero for outputs that are not shocked, making the
  Jacobian for het-outputs in `StageBlock` more robust.
- A keyword argument was not passed on to all calls of `f` in `Displace.apply`. Fix
  contributed by [Gregor Boehl](https://github.com/gboehl) (PR #8, upstream).

[Unreleased]: https://github.com/bbardoczy/sequence-jacobian/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/bbardoczy/sequence-jacobian/compare/v1.0.0...v1.1.0
