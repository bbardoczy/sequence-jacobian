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

import warnings
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=8)
def unit_circle(n_sample=8192):
    """The n_sample-th roots of unity, counterclockwise from z = 1.

    Every block must evaluate its symbol at the same points in the same order,
    since composing a DAG multiplies and adds them pointwise. Defined once here
    so that they cannot drift apart, and cached so that a DAG full of simple
    blocks builds it once rather than once per block. Blocks that reach for an
    FFT never need the points at all.

    The result is shared, so it is marked read-only. That stops the accidental
    routes -- in-place arithmetic raises, and views inherit the flag -- though a
    caller who deliberately calls setflags(write=True) can still corrupt it for
    everyone. Copy it if you need to modify it.

    Bounded rather than unbounded, since n_sample is in practice one value and
    an unbounded cache would retain an array per distinct value forever.
    Rebuilding costs tens of microseconds, so a miss is cheap.
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


def laurent_from_matrix(J):
    """Laurent coefficients read off the last column and row of a Jacobian.

    Appendix A.1. Entry m of the result is the coefficient on z**(m - (T-1)),
    taken as J[T-1+k, T-1] for k <= 0 and J[T-1, T-1-k] for k >= 0.

    Only valid where the corner is representative. That holds for a block that
    is exactly banded, and for a heterogeneous-agent Jacobian built by the fake
    news algorithm, which carries no truncation artifacts. It does NOT hold for
    a composed Jacobian: spurious missing anticipation contaminates exactly the
    corner these coefficients come from. Compose symbols instead of composing
    Jacobians and reading this.
    """
    return np.concatenate((J[:, -1], J[-1, :-1][::-1]))


class WindingNumber(int):
    """The winding number of det j(z), which reads as a verdict on H_U.

    An ordinary int -- it compares, indexes and serialises like one -- that
    explains itself when echoed. The reading, from propositions 9 and 11 of
    Auclert, Majic, Rognlie and Straub (2025):

        0   a unique bounded solution
        < 0 indeterminacy of dimension -w: bounded solutions form a family of
            that dimension, and one always exists
        > 0 nonexistence of dimension w: for generic shocks no bounded solution
            exists at all, though any that does exist is unique

    Take the null hypothesis to be that the model has a unique bounded solution.
    A non-zero winding number rejects it decisively. Zero fails to reject, but
    is only *generically* sufficient, so it can be a false positive: there are
    non-generic operators with winding number zero that are not invertible.

    That case is not exotic. The intertemporal Keynesian cross in goods space,
    I - M, is one: the household present-value budget constraint forces
    q'(I - M) = 0 for q = (1, (1+r)^-1, ...), so its range holds only sequences
    with zero net present value and it is not invertible -- yet its winding
    number is 0. Economic structure imposes exact restrictions that a randomly
    perturbed operator would never satisfy, so genericity in the space of
    operators does not transfer to the space of models. Suspect adding-up
    constraints and conservation laws, and use ``check_existence`` to rule the
    case out, since non-genericity always produces indeterminacy and
    nonexistence together.
    """

    def __repr__(self):
        if self == 0:
            reading = "unique bounded solution"
        elif self < 0:
            reading = (f"indeterminate: bounded solutions form a "
                       f"{-int(self)}-dimensional family")
        else:
            reading = (f"no bounded solution for generic shocks "
                       f"(nonexistence of dimension {int(self)})")
        return f"{int(self)}  ({reading})"


def winding_number_of_path(path, max_step=0.45 * np.pi):
    """Times a closed path in the complex plane encircles the origin.

    Accumulates the change in argument and divides by 2*pi. That is exact if the
    path is sampled finely enough for the argument to move by less than pi
    between neighbours; past that a turn is invisible and the count is wrong
    rather than merely imprecise.

    The default threshold sits just under pi/2 because a root within sampling
    distance of the circle turns the argument by about pi in total, and that
    turn arrives either as one step of nearly pi, when the root falls between
    samples, or as two of about pi/2, when it falls on one. Both have to trip.
    Models classified correctly leave room: the widest step seen across the
    shipped examples, including at the Taylor boundary itself, is 0.61 radians.
    """
    path = np.asarray(path)
    closed = np.append(path, path[0])

    steps = np.diff(np.angle(closed))
    steps = (steps + np.pi) % (2 * np.pi) - np.pi          # to (-pi, pi]

    biggest = np.max(np.abs(steps))
    if biggest > max_step:
        warnings.warn(
            f'The symbol turns by up to {biggest:.2f} radians between samples, so '
            'the winding number may miss a turn and cannot be trusted. Either '
            'raise n_sample, or -- if it does not help -- det j(z) has a zero on '
            'the unit circle, which the criterion assumes it does not. '
            + describe_near_zero(path))

    return WindingNumber(int(round(np.sum(steps) / (2 * np.pi))))


def describe_near_zero(path, n_sample=None):
    """Say where on the unit circle det j(z) comes closest to the origin.

    Worth reporting, because the location diagnoses what kind of root is nearly
    on the circle, and those point at different parts of a model. Where
    det j(z0) = 0, the sequence z0**(-s) is annihilated by H_U; it has modulus
    one, so it neither decays nor explodes, and is a bounded solution to the
    homogeneous system. z0 = 1 makes that sequence constant, which is a unit
    root -- something with no mean reversion, such as debt under a fiscal rule
    at its stability boundary, or wealth when beta * R = 1. z0 = -1 alternates.
    A complex z0 = exp(i*theta) oscillates at frequency theta, a deterministic
    cycle.
    """
    path = np.asarray(path)
    k = int(np.argmin(np.abs(path)))
    n = len(path) if n_sample is None else n_sample
    theta = 2 * np.pi * k / n

    if k == 0:
        return ('The closest approach is at z = 1, so the near-root is a unit '
                'root: some direction of the model has no mean reversion.')
    if 2 * k == n:
        return ('The closest approach is at z = -1, so the near-root alternates '
                'with period two.')
    return (f'The closest approach is at z = exp({theta:.3f}i), a near-root '
            f'oscillating with period {2 * np.pi / theta:.1f}.')
