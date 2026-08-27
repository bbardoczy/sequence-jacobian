"""Timings for .jacobian() and .solve_jacobian() on the shipped examples.

Deliberately not named test_*, so pytest does not collect it: these take a few
seconds, and wall-clock assertions in CI are flaky enough to be worse than
useless. Run it when you want to know, rather than on every commit:

    python tests/performance/bench_jacobian.py

To check whether a change costs anything, time each checkout and compare. A
scratch worktree keeps the comparison honest, since --src only prepends to
sys.path and the two trees must not otherwise differ:

    git worktree add ../ssj-base master
    python tests/performance/bench_jacobian.py --src ../ssj-base/src
    python tests/performance/bench_jacobian.py
    git worktree remove ../ssj-base

Compare the minimums, not the medians. On a machine doing anything else the
median picks up scheduling noise of several percent, while the minimum is the
closest thing to a clean run. Treat a difference under about 5% in the minimum
as nothing.

The two-asset household is the heaviest block in the package, so it is the one
to watch when changing the fake news algorithm; the HANK case is here because
it exercises solved blocks, and so the factored U_Z path that the others miss.
"""

import argparse
import statistics as st
import sys
import time


def bench(label, fn, reps):
    fn()  # warm up: numba compilation and any caches
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    print(f"  {label:38s} min {min(ts) * 1000:8.1f} ms   "
          f"median {st.median(ts) * 1000:8.1f} ms")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src', default=None,
                        help='sequence_jacobian source tree to time (default: importable one)')
    parser.add_argument('--reps', type=int, default=5)
    parser.add_argument('-T', type=int, default=300)
    args = parser.parse_args()

    if args.src is not None:
        sys.path.insert(0, args.src)

    import os

    import sequence_jacobian as sj
    from sequence_jacobian.examples import hank, krusell_smith, two_asset

    print(f"source: {os.path.dirname(sj.__file__)}")
    print(f"T = {args.T}, reps = {args.reps}\n")

    T, reps = args.T, args.reps

    _, ks_ss, ks_model, _, _, _ = krusell_smith.dag()
    ks_hh = ks_model['hh']
    bench("Krusell-Smith het block .jacobian",
          lambda: ks_hh.jacobian(ks_ss, ['r', 'w'], ['A', 'C'], T=T), reps)
    bench("Krusell-Smith full DAG .jacobian",
          lambda: ks_model.jacobian(ks_ss, ['K', 'Z'], ['asset_mkt'], T=T), reps)

    _, ta_ss, ta_model, ta_unknowns, ta_targets, ta_inputs = two_asset.dag()
    ta_hh = ta_model['hh']
    bench("two-asset het block .jacobian",
          lambda: ta_hh.jacobian(ta_ss, ['ra', 'rb', 'tax', 'w'], ['A', 'B', 'C'], T=T),
          reps)
    bench("two-asset .solve_jacobian",
          lambda: ta_model.solve_jacobian(ta_ss, ta_unknowns, ta_targets, ta_inputs, T=T),
          reps)

    _, hk_ss, hk_model, hk_unknowns, hk_targets, hk_inputs = hank.dag()
    bench("HANK .solve_jacobian",
          lambda: hk_model.solve_jacobian(hk_ss, hk_unknowns, hk_targets, hk_inputs, T=T),
          reps)


if __name__ == '__main__':
    main()
