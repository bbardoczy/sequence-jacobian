"""Determinacy in the sequence space.

Implements the winding number criterion of Auclert, Majic, Rognlie and Straub
(2025), "Determinacy and Large-Scale Solutions in the Sequence Space". H_U, the
Jacobian of targets with respect to unknowns, is a block quasi-Toeplitz operator
with a matrix-valued symbol j(z). Provided det j(z) has no zero on the unit
circle, the winding number of det j(z) about the origin says whether the model
has a unique bounded solution.

The symbol is built block by block and composed, never read off a composed
Jacobian. That is the paper's rule (appendix A.1): when a Jacobian is a product
or sum of others, evaluate the symbols of the parts and combine those. Reading
coefficients off the corner of a matrix is valid for a simple block, which is
exactly banded, and for a heterogeneous-agent Jacobian built by the fake news
algorithm, which carries no truncation artifacts -- but not for a composed
Jacobian, where spurious missing anticipation contaminates precisely the corner
the coefficients would come from.
"""

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=None)
def unit_circle(n_sample=8192):
    """The n_sample-th roots of unity, counterclockwise from z = 1.

    Every block must evaluate its symbol at the same points in the same order,
    since composing a DAG multiplies and adds them pointwise. Defined once here
    so that they cannot drift apart, and cached so that a DAG full of simple
    blocks builds it once rather than once per block. Blocks that reach for an
    FFT never need the points at all.

    The result is shared, so it is marked read-only: mutating it would corrupt
    every other holder.
    """
    z = np.exp(2j * np.pi * np.arange(n_sample) / n_sample)
    z.setflags(write=False)
    return z


def laurent_from_fake_news(F):
    """Laurent coefficients of a heterogeneous-agent block's symbol, from F.

    By proposition 3, the coefficient on z**k is the sum of the entries on the
    kth diagonal of the fake news matrix, meaning those with t - s = k.

    Taken from F rather than from the Jacobian on purpose. J accumulates F along
    its diagonals, so the last row and column of J hold only partial sums of
    these same quantities -- appendix A.1 uses them as an approximation, good
    when the block's internal persistence is short relative to T. Summing F is
    both exact and cheaper.

    Returns an array of length 2T-1, indexed so that entry m is the coefficient
    on z**(m - (T-1)).
    """
    T = F.shape[0]
    return np.array([np.trace(F, offset=-k) for k in range(-(T - 1), T)])


def evaluate_laurent(coefficients, n_sample):
    """Evaluate a Laurent polynomial on the n_sample-th roots of unity.

    Entry m of ``coefficients`` is the coefficient on z**(m - (T-1)). Placing
    each at index p modulo n_sample and taking an inverse DFT evaluates the
    polynomial at exp(2i*pi*k/n_sample) directly, in the same order and
    orientation as ``unit_circle``, with no sign or reversal corrections.

    n_sample must exceed the number of coefficients, or powers wrap onto each
    other and the result is aliased.
    """
    n = len(coefficients)
    if n_sample <= n:
        raise ValueError(f'Sampling a symbol with {n} coefficients at only '
                         f'{n_sample} points would alias; increase n_sample.')

    T = n // 2 + 1
    powers = np.arange(-(T - 1), T)

    padded = np.zeros(n_sample, dtype=complex)
    padded[powers % n_sample] = coefficients
    return np.fft.ifft(padded) * n_sample
