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
  `Block.impulse_nonlinear` no longer demeans internals. This changes returned numbers
  rather than raising an error, so results computed with 1.0.0 that rely on demeaned
  internals will differ.
- `StageBlock._impulse_nonlinear` returns all internals.
- Applying a `Bijection` to a `dict` now prioritises remapped names when a key clash
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
- `curlyY` is now initialised to zero for outputs that are not shocked, making the
  Jacobian for het-outputs in `StageBlock` more robust.
- A keyword argument was not passed on to all calls of `f` in `simple_displacement`.

[Unreleased]: https://github.com/bbardoczy/sequence-jacobian/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/bbardoczy/sequence-jacobian/compare/v1.0.0...v1.1.0
