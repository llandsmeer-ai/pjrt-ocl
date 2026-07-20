"""Offline phase-count instrument: lower a bench_transformer/decode config to a
VMProgram, build the schedule levels, and print the number of compute phases
(= global barriers) plus a per-tile-op census. Headline metric for §33.

Usage:  python tools/phase_count.py --config base [--decode]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base")
    ap.add_argument("--decode", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax, jax.numpy as jnp
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    import bench_transformer as BT

    cfg = BT.CONFIGS[args.config]
    model = BT.make_model(jnp)
    x, params = BT.make_params(cfg)
    B, T, D, H, F, L_ = cfg
    if args.decode:
        # single-token decode step (no KV cache modelled; matvec shapes)
        x = x[:, :1, :]
    jx = jnp.asarray(x)
    jp = [{k: jnp.asarray(v) for k, v in p.items()} for p in params]
    fn = jax.jit(lambda x, p: model(x, p, H))
    lowered = fn.lower(jx, jp)
    artifact = lowered.compiler_ir("stablehlo")
    # serialize to VHLO portable artifact bytes
    import jaxlib.mlir.dialects.stablehlo as sh
    mod_bytes = sh.serialize_portable_artifact(
        artifact, sh.get_current_version())

    prog = _lower(mod_bytes)
    n_compute, n_while, census, op_census, names = _phase_census(prog)
    print(f"config={args.config} decode={args.decode} "
          f"phases(compute)={n_compute} while={n_while} "
          f"instrs(main)={prog.main_len}")
    print("  tile-op census (per-phase task count):",
          {names.get(k, k): v for k, v in sorted(census.items())})
    top = {k: v for k, v in op_census.most_common(12)}
    print("  op census:", top)


if __name__ == "__main__":
    main()
