# Sequence-Space Jacobian (SSJ) — Bardóczy fork

[![CI](https://github.com/bbardoczy/sequence-jacobian/actions/workflows/main.yml/badge.svg)](https://github.com/bbardoczy/sequence-jacobian/actions/workflows/main.yml)

SSJ is a toolkit for analyzing dynamic macroeconomic models with (or without) rich
microeconomic heterogeneity.

**This is a personal fork of [shade-econ/sequence-jacobian](https://github.com/shade-econ/sequence-jacobian).**
I maintain it for my own research and for the replication files that go with it, which
means it is a place to try things out without waiting on consensus. Some of what is here
may eventually be proposed upstream; some of it exists because it suits how I work.
Nothing here is endorsed by the upstream development team.

If you want the canonical toolkit — the one the paper describes, the one that is on PyPI,
the one that gets support — use [shade-econ/sequence-jacobian](https://github.com/shade-econ/sequence-jacobian)
instead. Come here only if you specifically want something below.

## What this fork adds

Version numbers are independent of upstream and start from upstream `v1.0.0`;
[CHANGELOG.md](CHANGELOG.md) records every difference, release by release.

- **Determinacy checking** by the winding number criterion, so a model can be asked
  directly whether it has a unique bounded solution. `Block.winding_number` returns the
  verdict and `Block.check_existence` rules out the one case where that verdict can
  mislead. See the [determinacy notebook](notebooks/determinacy.ipynb).
- **The fake news matrix is public.** `HetBlock.fake_news_matrix()` returns the object the
  Jacobian is assembled from, which is worth inspecting on its own.
- **Blocks and their Jacobians are named consistently**, and anonymous or duplicated names
  are refused rather than silently resolved. This fixed cases where a Jacobian was quietly
  applied to the wrong block, or a block was dropped from addressing while still evaluating.
- **A reproducible development environment** in [environment.yml](environment.yml), and a
  ruff configuration pinned to the Python this package actually supports.

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

*Optional package*: there is an optional interface for plotting the directed acyclic graph
(DAG) representation of models, which requires
[Graphviz for Python](https://github.com/xflr6/graphviz#graphviz). It is already in
`environment.yml`; otherwise `conda install -c conda-forge python-graphviz`.

## Introductory notebooks

The notebooks below are the best way to learn the toolkit, and are meant to be worked
through in order. The first five come from upstream.

- [RBC](https://github.com/bbardoczy/sequence-jacobian/blob/master/notebooks/rbc.ipynb)
    - represent macro models as collections of blocks (DAG)
    - write SimpleBlocks and CombinedBlocks
    - compute linearized and non-linear (perfect-foresight) impulse responses
- [Krusell-Smith](https://github.com/bbardoczy/sequence-jacobian/blob/master/notebooks/krusell_smith.ipynb)
    - write HetBlocks to represent heterogeneous agents
    - construct general-equilibrium Jacobians manually
    - compute the log-likelihood of the model given time-series data
- [One-asset HANK](https://github.com/bbardoczy/sequence-jacobian/blob/master/notebooks/hank.ipynb)
    - adapt an off-the-shelf HetBlock to any macro environment using helper functions
    - see a more advanced example of calibration
- [Two-asset HANK](https://github.com/bbardoczy/sequence-jacobian/blob/master/notebooks/two_asset.ipynb)
    - write SolvedBlocks to represent implicit aggregate equilibrium conditions
    - re-use saved Jacobians
    - fine tune options of block methods
- [Labor search](https://github.com/bbardoczy/sequence-jacobian/blob/master/notebooks/labor_search.ipynb)
    - example with multiple exogenous states
    - shocks to transition matrix of exogenous states
- [Determinacy](https://github.com/bbardoczy/sequence-jacobian/blob/master/notebooks/determinacy.ipynb) *(this fork only)*
    - read the symbol of a block, and plot det j(z) around the unit circle
    - test whether a model has a unique bounded solution, via the winding number
    - recover the Taylor principle as a determinacy boundary
    - rule out the false positives with `check_existence`

## Attribution

The conceptual framework, and the great majority of the code here, are the work of the
upstream SSJ team. The method is described in Adrien Auclert, Bence Bardóczy, Matthew
Rognlie, Ludwig Straub (2021),
[Using the Sequence-Space Jacobian to Solve and Estimate Heterogeneous-Agent Models](https://doi.org/10.3982/ECTA17434),
Econometrica 89(5), pp. 2375–2408
[[ungated copy]](http://mattrognlie.com/sequence_space_jacobian.pdf). The upstream
development team is Bence Bardóczy, Michael Cai and Matthew Rognlie, with contributions
from Adrien Auclert, Martin Souchier and Ludwig Straub.

The determinacy code implements Adrien Auclert, Ivan Majic, Matthew Rognlie, Ludwig Straub
(2025), [Determinacy and Large-Scale Solutions in the Sequence Space](https://web.stanford.edu/~aauclert/determinacy_sequence_space.pdf).

This fork is maintained by Bence Bardóczy. Anything broken in it is my doing, not
upstream's — please raise it on
[this fork's issue tracker](https://github.com/bbardoczy/sequence-jacobian/issues) rather
than theirs.

## Resources

If you'd like to learn more about Python, its numerical libraries, and Jupyter notebooks,
the [introductory lectures at QuantEcon](https://python-programming.quantecon.org/intro.html)
are a terrific place to start. More advanced tutorials for numerical Python include the
[SciPy Lecture Notes](http://scipy-lectures.org/intro/language/python_language.html) and
the [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/).

For those who used the pre-1.0 toolkit, which differed in a number of ways, upstream keeps
an [early toolkit page](https://github.com/shade-econ/sequence-jacobian/tree/bcca2eff6041abc77d0a777e6c64f9ac6ff44305).
