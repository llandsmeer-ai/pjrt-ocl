"""Dump the per-phase task list (tile op, tile count, element count) that the
scheduler builds for a bench-suite workload — the offline view of what the VM
will actually launch.

Usage:
    . ./env.sh
    JAX_PLATFORMS=cpu .venv/bin/python tools/dump_tasks.py --name monte_carlo
"""
from __future__ import annotations

import argparse
import collections
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--full", action="store_true", help="list every phase")
    a = ap.parse_args()
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    import jax.numpy as jnp
    import jaxlib.mlir.dialects.stablehlo as sh
    from bench_suite import workloads
    from pjrt_ocl import lowering as L
    from pjrt_ocl import scheduler as S

    fn, tree, meta = workloads.build(a.name)
    tree_j = jax.tree_util.tree_map(jnp.asarray, tree)
    art = jax.jit(fn).lower(tree_j).compiler_ir("stablehlo")
    prog = L.lower_artifact(sh.serialize_portable_artifact(
        art, sh.get_current_version()))
    nlanes = int(os.environ.get("PJRT_OCL_NLANES", "32"))
    sc = S._Scheduler(prog, S.DeviceConfig.from_env(), nlanes)
    levels = sc._build_levels(list(range(prog.main_len)))
    names = {getattr(S, n): n for n in dir(S) if n.startswith("TILE_")}

    tot_tiles = collections.Counter()
    tot_cost = collections.Counter()
    nph = 0
    for kind, payload in levels:
        if kind != "compute":
            continue
        nph += 1
        row = []
        for idx in payload:
            t = sc.tasks[sc._task_for(idx)]
            nt = t.n_tiles()
            tot_tiles[names.get(t.tile_op, t.tile_op)] += nt
            tot_cost[names.get(t.tile_op, t.tile_op)] += 1
            row.append(f"{names.get(t.tile_op,t.tile_op)[5:]}"
                       f"(tiles={nt},p0={t.p0},p1={t.p1},p2={t.p2},p3={t.p3})")
        if a.full:
            print(f"  phase {nph:3d}: " + "  ".join(row))
    print(f"{a.name}: phases={nph} tasks={sum(tot_cost.values())}")
    for k in sorted(tot_cost, key=lambda k: -tot_tiles[k]):
        print(f"   {k:<22} tasks={tot_cost[k]:5d} total_tiles={tot_tiles[k]:7d}")


if __name__ == "__main__":
    main()
