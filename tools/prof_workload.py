"""Per-workload profiler for the bench-suite workloads (investigation tool).

For ONE workload from tools/bench_suite/workloads.py it reports, on the target
backend:

  * end-to-end median-of-5 latency (same methodology as run_suite.py),
  * a "dispatch floor" control: the SAME pytree of inputs fed through an
    identity-shaped jit (returns a tiny slice of each input summed), so the
    per-Execute + per-I/O-buffer floor of decisions.md §49 is measured for this
    workload's exact argument count/shapes,
  * PJRT_OCL_PROFILE / PJRT_OCL_PHASE_STATS pass-through (set them in env),
  * offline phase census (number of compute phases = barriers, tile-op mix)
    when --phases is given (lowers with pjrt_ocl.lowering on the CPU backend).

Usage:
    . ./env.sh
    JAX_PLATFORMS=opencl PJRT_OCL_DEVICE=Intel \
      .venv/bin/python tools/prof_workload.py --name nbody
    JAX_PLATFORMS=cpu .venv/bin/python tools/prof_workload.py --name nbody
    .venv/bin/python tools/prof_workload.py --name nbody --phases   # offline
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))


def _time(fn_j, tree_j, jax, budget=0.08):
    for _ in range(3):
        jax.block_until_ready(fn_j(tree_j))
    t0 = time.perf_counter()
    jax.block_until_ready(fn_j(tree_j))
    one = time.perf_counter() - t0
    iters = max(3, min(200, int(budget / max(one, 1e-6))))
    ts = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(iters):
            r = fn_j(tree_j)
        jax.block_until_ready(r)
        ts.append((time.perf_counter() - t) / iters)
    ts.sort()
    return ts[len(ts) // 2] * 1e3, iters


def run_bench(name, floor=True, reps=1):
    import numpy as np
    import jax
    import jax.numpy as jnp
    from bench_suite import workloads

    fn, tree, meta = workloads.build(name)
    tree_j = jax.tree_util.tree_map(jnp.asarray, tree)
    fn_j = jax.jit(fn)
    leaves = jax.tree_util.tree_leaves(tree_j)
    out = fn_j(tree_j)
    jax.block_until_ready(out)
    outs = jax.tree_util.tree_leaves(out)
    nbytes_in = sum(int(np.prod(l.shape)) * l.dtype.itemsize for l in leaves)
    nbytes_out = sum(int(np.prod(l.shape)) * l.dtype.itemsize for l in outs)

    best = None
    for _ in range(reps):
        ms, iters = _time(fn_j, tree_j, jax)
        best = ms if best is None else min(best, ms)
    print(f"{name}: {best:.4f} ms  (iters={iters}, n_in={len(leaves)} "
          f"{nbytes_in/1024:.1f} KiB, n_out={len(outs)} {nbytes_out/1024:.1f} KiB)")

    if floor:
        # Same argument count + shapes, ~zero compute: sum of first element of
        # every leaf. Measures this workload's dispatch floor.
        def ident(p):
            acc = None
            for l in jax.tree_util.tree_leaves(p):
                v = l.reshape(-1)[0].astype(jnp.float32)
                acc = v if acc is None else acc + v
            return acc
        f2 = jax.jit(ident)
        ms2, _ = _time(f2, tree_j, jax)
        print(f"{name}: FLOOR (same {len(leaves)} inputs, 1 scalar out) "
              f"{ms2:.4f} ms  => compute-above-floor {best - ms2:.4f} ms")
    return best


def phase_census(name):
    """Offline: lower the workload to a VMProgram and count compute phases."""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    import numpy as np
    from bench_suite import workloads
    from pjrt_ocl import lowering as L
    from pjrt_ocl import scheduler as S

    fn, tree, meta = workloads.build(name)
    import jax.numpy as jnp
    import jaxlib.mlir.dialects.stablehlo as sh
    tree_j = jax.tree_util.tree_map(jnp.asarray, tree)
    lowered = jax.jit(fn).lower(tree_j)
    artifact = lowered.compiler_ir("stablehlo")
    mod_bytes = sh.serialize_portable_artifact(artifact, sh.get_current_version())
    prog = L.lower_artifact(mod_bytes)
    nlanes = int(os.environ.get("PJRT_OCL_NLANES", "32"))
    sc = S._Scheduler(prog, S.DeviceConfig.from_env(), nlanes)
    levels = sc._build_levels(list(range(prog.main_len)))
    n_compute = sum(1 for k, _ in levels if k == "compute")
    n_other = len(levels) - n_compute
    census = collections.Counter()
    for kind, payload in levels:
        if kind == "compute":
            for idx in payload:
                census[sc.tasks[sc._task_for(idx)].tile_op] += 1
    names = {getattr(S, n): n for n in dir(S) if n.startswith("TILE_")}
    op_census = collections.Counter()
    for ins in prog.instrs[:prog.main_len]:
        op_census[L.OP_NAMES.get(ins.op, ins.op)] += 1
    print(f"{name}: instrs={prog.main_len} compute_phases={n_compute} "
          f"other_levels={n_other}")
    print("  tile ops:", {names.get(k, k): v for k, v in census.most_common()})
    print("  vm ops:  ", dict(op_census.most_common()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--phases", action="store_true")
    ap.add_argument("--no-floor", action="store_true")
    ap.add_argument("--reps", type=int, default=1)
    a = ap.parse_args()
    if a.phases:
        phase_census(a.name)
    else:
        run_bench(a.name, floor=not a.no_floor, reps=a.reps)


if __name__ == "__main__":
    main()
