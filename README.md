# Sequence-Space Jacobian (SSJ) — Bence's fork

[![CI](https://github.com/bbardoczy/sequence-jacobian/actions/workflows/main.yml/badge.svg)](https://github.com/bbardoczy/sequence-jacobian/actions/workflows/main.yml)

SSJ is a toolkit for analyzing dynamic macroeconomic models with (or without) rich
microeconomic heterogeneity.

**This is a personal fork of [shade-econ/sequence-jacobian](https://github.com/shade-econ/sequence-jacobian).**
I maintain it for my own research and for the replication files that go with it. Some of what is here
may eventually be proposed upstream; some of it exists because it suits my work. I am solely responsible for the changes and any mistakes made here.

## What this fork adds

Version numbers are independent of upstream and start from upstream `v1.0.0`;
[CHANGELOG.md](CHANGELOG.md) records every difference, release by release.

- **Determinacy checking** *(1.2)* by the winding number criterion, so a model can be asked
  directly whether it has a unique bounded solution. `Block.winding_number` returns the
  verdict and `Block.check_existence` rules out the one case where that verdict can
  mislead. See the [determinacy notebook](notebooks/determinacy.ipynb).
- **`ConditionalMarkov`** *(1.1)*, a `LawOfMotion` whose transition matrix may depend on the
  other states rather than only on the one it updates. `ExogenousMaker` accepts these
  alongside ordinary Markov matrices.
- **Quality-of-life improvements** *(1.1, 1.2)*. The ones most likely to matter:
    - blocks and the Jacobians they produce are named consistently, and anonymous or
      duplicated names are refused rather than silently resolved *(1.2)*. This fixed cases
      where a Jacobian was quietly applied to the wrong block, or a block was dropped from
      addressing while still evaluating.
    - `HetBlock.fake_news_matrix()` exposes the object the Jacobian is assembled from, which
      is worth inspecting on its own *(1.2)*
    - arithmetic on `ImpulseDict` no longer applies to `internals` *(1.1)*. This changes
      returned numbers rather than raising, so results computed under 1.0 that relied on
      demeaned internals will differ.
    - `StageBlock` can report a narrowed set of internals instead of always returning
      everything *(1.1)*
- **A reproducible development environment** *(1.1)* in [environment.yml](environment.yml).

## Requirements and installation

SSJ runs on Python 3.9 or newer and requires NumPy, SciPy and Numba.

Note that `pip install sequence-jacobian` installs **upstream**, not this fork. To install
this one:

```
pip install git+https://github.com/bbardoczy/sequence-jacobian.git
```

To work on it, clone the repository and build the development environment, which installs
the package in editable mode:

```
mamba env create -f environment.yml
conda activate ssj-develop
pytest
```

## Introductory notebooks

The notebooks below provide an introduction into the toolkit, and are meant to be worked
through in order.

1. [RBC](https://github.com/bbardoczy/sequence-jacobian/blob/master/notebooks/rbc.ipynb)
    - represent macro models as collections of blocks (DAG)
    - write SimpleBlocks and CombinedBlocks
    - compute linearized and non-linear (perfect-foresight) impulse responses
2. [Krusell-Smith](https://github.com/bbardoczy/sequence-jacobian/blob/master/notebooks/krusell_smith.ipynb)
    - write HetBlocks to represent heterogeneous agents
    - construct general-equilibrium Jacobians manually
    - compute the log-likelihood of the model given time-series data
3. [One-asset HANK](https://github.com/bbardoczy/sequence-jacobian/blob/master/notebooks/hank.ipynb)
    - adapt an off-the-shelf HetBlock to any macro environment using helper functions
    - see a more advanced example of calibration
4. [Two-asset HANK](https://github.com/bbardoczy/sequence-jacobian/blob/master/notebooks/two_asset.ipynb)
    - write SolvedBlocks to represent implicit aggregate equilibrium conditions
    - re-use saved Jacobians
    - fine tune options of block methods
5. [Labor search](https://github.com/bbardoczy/sequence-jacobian/blob/master/notebooks/labor_search.ipynb)
    - example with multiple exogenous states
    - shocks to transition matrix of exogenous states
6. [Determinacy](https://github.com/bbardoczy/sequence-jacobian/blob/master/notebooks/determinacy.ipynb) *(this fork only)*
    - read the symbol of a block, and plot det j(z) around the unit circle
    - test whether a model has a unique bounded solution, via the winding number
    - rule out the false positives with `check_existence`

## Attribution

The SSJ framework was introduced by Adrien Auclert, Bence Bardóczy, Matthew
Rognlie, Ludwig Straub (2021),
[Using the Sequence-Space Jacobian to Solve and Estimate Heterogeneous-Agent Models](https://doi.org/10.3982/ECTA17434),
Econometrica 89(5), pp. 2375–2408
[[ungated copy]](http://mattrognlie.com/sequence_space_jacobian.pdf).

The determinacy code implements Adrien Auclert, Ivan Majic, Matthew Rognlie, Ludwig Straub
(2025), [Determinacy and Large-Scale Solutions in the Sequence Space](https://web.stanford.edu/~aauclert/determinacy_sequence_space.pdf).

The great majority of the code is the work of the upstream
development team: Matthew Rognlie, Bence Bardóczy, and Michael Cai, with contributions
from Adrien Auclert, Martin Souchier and Ludwig Straub.

