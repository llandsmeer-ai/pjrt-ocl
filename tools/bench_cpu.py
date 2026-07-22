"""CPU bench: ours-on-PoCL (opencl/Portable) vs native JAX CPU, separate procs.

Usage:
    . ./env.sh
    .venv/bin/python tools/bench_cpu.py --only layernorm batchnorm heat2d nbody logistic_map
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, tempfile, time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
SENTINEL = "##BENCHJSON##"

BACKENDS = {
    "pocl": {"JAX_PLATFORMS": "opencl", "PJRT_OCL_DEVICE": "Portable"},
    "cpu":  {"JAX_PLATFORMS": "cpu"},
}


def _worker(name, outdir):
    import numpy as np
    from bench_suite import workloads
    result = {"name": name}
    import jax, jax.numpy as jnp
    fn, tree, meta = workloads.build(name)
    result["meta"] = meta
    tree_j = jax.tree_util.tree_map(lambda a: jnp.asarray(a), tree)
    fn_j = jax.jit(fn)
    out = fn_j(tree_j); jax.block_until_ready(out)
    out_np = np.asarray(out)
    np.save(os.path.join(outdir, name + ".npy"), out_np)
    result.update(status="PASS", shape=list(out_np.shape),
                  mean=float(out_np.mean()), std=float(out_np.std()))
    for _ in range(3):
        jax.block_until_ready(fn_j(tree_j))
    t0 = time.perf_counter(); jax.block_until_ready(fn_j(tree_j))
    one = time.perf_counter() - t0
    iters = max(3, min(500, int(0.1 / max(one, 1e-6))))
    ts = []
    for _ in range(7):
        t = time.perf_counter()
        for _ in range(iters):
            r = fn_j(tree_j)
        jax.block_until_ready(r)
        ts.append((time.perf_counter() - t) / iters)
    ts.sort()
    result["ms"] = float(ts[len(ts) // 2]) * 1e3
    result["ms_min"] = float(ts[0]) * 1e3
    result["iters"] = iters
    print(SENTINEL + json.dumps(result))
    return 0


def _run(name, backend, outdir, timeout=360):
    env = dict(os.environ); env.update(BACKENDS[backend])
    cmd = [sys.executable, os.path.abspath(__file__), "--worker",
           "--name", name, "--outdir", outdir]
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"name": name, "status": "TIMEOUT"}
    for line in p.stdout.splitlines():
        if line.startswith(SENTINEL):
            return json.loads(line[len(SENTINEL):])
    tail = (p.stderr.strip().splitlines() or ["(no stderr)"])[-3:]
    return {"name": name, "status": "CRASH", "error": "\n".join(tail)}


def main():
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--name"); ap.add_argument("--outdir")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    if args.worker:
        return _worker(args.name, args.outdir)
    from bench_suite import workloads
    names = args.only or ["layernorm", "batchnorm", "heat2d", "nbody", "logistic_map"]
    outdir = tempfile.mkdtemp(prefix="benchcpu_")
    print(f"{'workload':16} {'ours(PoCL)':>12} {'JAX-CPU':>10} {'gap':>7}  correct")
    for name in names:
        o = _run(name, "pocl", outdir)
        src = os.path.join(outdir, name + ".npy")
        if os.path.exists(src): os.replace(src, os.path.join(outdir, name + ".ours.npy"))
        c = _run(name, "cpu", outdir)
        cok = "?"
        of = os.path.join(outdir, name + ".ours.npy"); cf = os.path.join(outdir, name + ".npy")
        if os.path.exists(of) and os.path.exists(cf):
            a, b = np.load(of), np.load(cf)
            if a.shape == b.shape:
                cok = "EXACT" if np.array_equal(a, b) else (
                    "close" if np.allclose(a, b, atol=1e-5, rtol=1e-5) else
                    f"DIFF {np.abs(a-b).max():.2e}")
        om = o.get("ms"); cm = c.get("ms")
        gap = f"{om/cm:.2f}x" if (om and cm) else "-"
        os_ = o.get("status"); cs_ = c.get("status")
        oms = f"{om:.3f}ms" if om else os_
        cms = f"{cm:.3f}ms" if cm else cs_
        print(f"{name:16} {oms:>12} {cms:>10} {gap:>7}  {cok}")
        if os_ != "PASS": print(f"    ours: {o.get('error','')[:200]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
