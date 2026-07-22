"""Offline phase-count instrument: lower a program to a VMProgram, build the
schedule levels, and print the number of compute phases (= global barriers)
plus a per-tile-op census. Headline metric for §33.

Two sources of programs:

  --config base [--decode]      tools/bench_transformer.py configs (original use)
  --workload rk4_ode            tools/bench_suite/workloads.py entries (§52)

For loop workloads the census RECURSES into region sub-lists (FOR/WHILE/IF
bodies) and reports phases-per-iteration x trip count, which is the number
that actually prices a host-dispatch run (one clEnqueueNDRangeKernel per
phase).

Usage:
    python tools/phase_count.py --config base [--decode]
    python tools/phase_count.py --workload rk4_ode [--verbose]
"""
from __future__ import annotations
import argparse, collections, os, sys

import numpy as np


def _lower(artifact):
    from pjrt_ocl import lowering as L
    return L.lower_artifact(artifact)


def _phase_census(prog):
    from pjrt_ocl import lowering as L
    from pjrt_ocl import scheduler as S
    sc = S._Scheduler(prog, S.DeviceConfig.from_env(), 1)
    levels = sc._build_levels(list(range(prog.main_len)))
    n_compute = 0
    n_while = 0
    census = collections.Counter()
    op_census = collections.Counter()
    for kind, payload in levels:
        if kind == "compute":
            n_compute += 1
            for idx in payload:
                t = sc.tasks[sc._task_for(idx)]
                census[t.tile_op] += 1
        else:
            n_while += 1
    for ins in prog.instrs[:prog.main_len]:
        op_census[L.OP_NAMES.get(ins.op, ins.op)] += 1
    names = {getattr(S, n): n for n in dir(S) if n.startswith("TILE_")}
    return n_compute, n_while, census, op_census, names


# --------------------------------------------------------------- recursive ---

def _walk(sc, indices, depth, out, verbose):
    """Recursively count phases over an instruction sub-list. Appends
    (depth, kind, detail) rows to `out` and returns (n_phases, n_dispatch)
    where n_dispatch weights a region's phases by its trip count."""
    from pjrt_ocl import lowering as L
    levels = sc._build_levels(list(indices))
    n_phases = 0
    n_dispatch = 0
    for kind, payload in levels:
        if kind == "compute":
            n_phases += 1
            n_dispatch += 1
            ops = []
            for idx in sorted(payload):
                ins = sc.prog.instrs[idx]
                ops.append(f"{L.OP_NAMES.get(ins.op, hex(ins.op))}(n={ins.n})")
            out.append((depth, "phase", ", ".join(ops)))
        else:
            ins = sc.prog.instrs[payload]
            if ins.op == L.OP_FOR:
                trip = ins.b
                out.append((depth, "FOR", f"trip={trip} body=[{ins.n},{ins.n + ins.imm})"))
                bp, bd = _walk(sc, range(ins.n, ins.n + ins.imm), depth + 1,
                               out, verbose)
                out.append((depth, "FOR-end",
                            f"body phases/iter={bp}  x{trip} = {bp * trip}"))
                n_phases += bp
                n_dispatch += bd * trip
            elif ins.op == L.OP_WHILE:
                out.append((depth, "WHILE",
                            f"cond=[{ins.a},{ins.a + ins.b}) "
                            f"body=[{ins.n},{ins.n + ins.imm})"))
                cp, cd = _walk(sc, range(ins.a, ins.a + ins.b), depth + 1,
                               out, verbose)
                bp, bd = _walk(sc, range(ins.n, ins.n + ins.imm), depth + 1,
                               out, verbose)
                out.append((depth, "WHILE-end",
                            f"cond phases/iter={cp} body phases/iter={bp} "
                            f"(+1 blocking cond readback/iter)"))
                n_phases += cp + bp
                n_dispatch += cd + bd     # trip count unknown at compile time
            else:                          # OP_IF
                out.append((depth, "IF", "then/else"))
                tp, td = _walk(sc, range(ins.a, ins.a + ins.b), depth + 1,
                               out, verbose)
                ep, ed = _walk(sc, range(ins.n, ins.n + ins.imm), depth + 1,
                               out, verbose)
                n_phases += tp + ep
                n_dispatch += max(td, ed)
    return n_phases, n_dispatch


def _deep_census(prog, verbose=False):
    from pjrt_ocl import scheduler as S
    sc = S._Scheduler(prog, S.DeviceConfig.from_env(),
                      S.DeviceConfig.from_env().nlanes)
    out = []
    n_phases, n_dispatch = _walk(sc, range(prog.main_len), 0, out, verbose)
    return n_phases, n_dispatch, out


# ----------------------------------------------------------------- drivers ---

def _serialize(lowered):
    import jaxlib.mlir.dialects.stablehlo as sh
    artifact = lowered.compiler_ir("stablehlo")
    return sh.serialize_portable_artifact(artifact, sh.get_current_version())


def _run_transformer(args):
    import jax, jax.numpy as jnp
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    import bench_transformer as BT

    cfg = BT.CONFIGS[args.config]
    model = BT.make_model(jnp)
    x, params = BT.make_params(cfg)
    B, T, D, H, F, L_ = cfg
    if args.decode:
        x = x[:, :1, :]
    jx = jnp.asarray(x)
    jp = [{k: jnp.asarray(v) for k, v in p.items()} for p in params]
    fn = jax.jit(lambda x, p: model(x, p, H))
    mod_bytes = _serialize(fn.lower(jx, jp))

    prog = _lower(mod_bytes)
    n_compute, n_while, census, op_census, names = _phase_census(prog)
    print(f"config={args.config} decode={args.decode} "
          f"phases(compute)={n_compute} while={n_while} "
          f"instrs(main)={prog.main_len}")
    print("  tile-op census (per-phase task count):",
          {names.get(k, k): v for k, v in sorted(census.items())})
    top = {k: v for k, v in op_census.most_common(12)}
    print("  op census:", top)


def _run_workload(name, verbose=False):
    import jax, jax.numpy as jnp
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from bench_suite import workloads as W

    fn, tree, meta = W.build(name)
    jtree = jax.tree.map(lambda a: jnp.asarray(a), tree)
    mod_bytes = _serialize(jax.jit(fn).lower(jtree))
    prog = _lower(mod_bytes)
    n_phases, n_dispatch, rows = _deep_census(prog, verbose)
    print(f"=== {name}: static phases={n_phases}  "
          f"DISPATCHES(trip-weighted)={n_dispatch}  "
          f"instrs={len(prog.instrs)} (main {prog.main_len})")
    for depth, kind, detail in rows:
        if not verbose and kind == "phase" and depth == 0:
            # root-level phases are one-off; summarize only in verbose mode
            pass
        print("   " + "  " * depth + f"[{kind}] {detail}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--decode", action="store_true")
    ap.add_argument("--workload", nargs="*", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    if args.workload:
        for nm in args.workload:
            _run_workload(nm, args.verbose)
            print()
        return
    args.config = args.config or "base"
    _run_transformer(args)


if __name__ == "__main__":
    main()
