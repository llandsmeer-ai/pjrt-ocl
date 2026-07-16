"""poc/12: fixed-trip while — performance comparison of the three loop paths.

Measures the SAME jax programs through the real plugin with PJRT_OCL_WHILE
pinned to while / for / unroll (subprocess per config: the lowering cache and
plugin state are per-process). Baseline: jax CPU backend in-process.

Workloads:
  fori-ew   : x = x*a + b (vector a,b — not scalar-foldable) for T steps, size n
  scan-rnn  : c = c*0.9 + xs[t], stacking ys (dynamic_slice + DUS per step)

Run:  .venv/bin/python poc/12-fixed-trip/bench.py [--device Portable|NVIDIA]
Emits one CSV row per (workload, n, T, mode): best-of-5 wall ms.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

CHILD = r"""
import json, os, sys, time
import numpy as np
import jax, jax.numpy as jnp

cfg = json.loads(sys.argv[1])
if cfg["backend"] == "opencl":
    jax.config.update("jax_platforms", "opencl")

def fori_ew(n, T):
    a = jnp.linspace(0.9, 1.1, n, dtype=jnp.float32)
    b = jnp.linspace(-0.1, 0.1, n, dtype=jnp.float32)
    def f(x):
        return jax.lax.fori_loop(0, T, lambda i, x: x * a + b, x)
    x = jnp.zeros((n,), jnp.float32) + 0.5
    return f, (x,)

def scan_rnn(n, T):
    def f(c, xs):
        def step(c, xt):
            c = c * 0.9 + xt
            return c, c
        return jax.lax.scan(step, c, xs)
    c = jnp.zeros((n,), jnp.float32)
    xs = jnp.ones((T, n), jnp.float32) * 0.01
    return f, (c, xs)

fn, args = {"fori-ew": fori_ew, "scan-rnn": scan_rnn}[cfg["work"]](
    cfg["n"], cfg["T"])
jf = jax.jit(fn)
t_compile0 = time.perf_counter()
out = jf(*args)
jax.block_until_ready(out)
t_compile = time.perf_counter() - t_compile0   # includes first execute

best = float("inf")
for _ in range(cfg["reps"]):
    t0 = time.perf_counter()
    jax.block_until_ready(jf(*args))
    dt = time.perf_counter() - t0
    best = min(best, dt)
flat = jax.tree_util.tree_leaves(out)
sig = float(np.abs(np.asarray(flat[0], dtype=np.float32)).mean())
print(json.dumps({"ms": best * 1e3, "compile_s": t_compile, "sig": sig}))
"""


def run_one(work: str, n: int, T: int, backend: str, mode: str,
            device: str, reps: int = 5) -> dict:
    env = dict(os.environ)
    env["PJRT_OCL_WHILE"] = mode
    env["PJRT_OCL_DEVICE"] = device
    cfg = {"work": work, "n": n, "T": T, "backend": backend, "reps": reps}
    try:
        p = subprocess.run(
            [sys.executable, "-c", CHILD, json.dumps(cfg)],
            capture_output=True, text=True, env=env, cwd=ROOT, timeout=300)
    except subprocess.TimeoutExpired:
        return {"error": "TIMEOUT"}
    if p.returncode != 0:
        lines = [l for l in (p.stderr or "").strip().splitlines()
                 if "experimental" not in l] or [f"exit {p.returncode}"]
        return {"error": lines[-1][:120]}
    return json.loads(p.stdout.strip().splitlines()[-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="Portable",
                    help="PJRT_OCL_DEVICE platform substring")
    ap.add_argument("--sizes", default="4096,1048576")
    ap.add_argument("--trips", default="8,32,128,512")
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]
    trips = [int(t) for t in args.trips.split(",")]

    print(f"# device={args.device}")
    print("work,n,T,mode,ms,compile_s,sig")
    for work in ("fori-ew", "scan-rnn"):
        for n in sizes:
            for T in trips:
                if work == "scan-rnn" and n * T > 32 << 20:
                    continue           # xs would exceed ~128 MB
                rows = {}
                ref = run_one(work, n, T, "cpu", "auto", args.device, args.reps)
                rows["xla-cpu"] = ref
                for mode in ("while", "for", "unroll"):
                    rows[mode] = run_one(work, n, T, "opencl", mode,
                                         args.device, args.reps)
                for mode, r in rows.items():
                    if "error" in r:
                        print(f"{work},{n},{T},{mode},ERROR,,{r['error']}")
                    else:
                        sig = r["sig"]
                        print(f"{work},{n},{T},{mode},{r['ms']:.3f},"
                              f"{r['compile_s']:.2f},{sig:.5f}")
                sys.stdout.flush()


if __name__ == "__main__":
    main()
