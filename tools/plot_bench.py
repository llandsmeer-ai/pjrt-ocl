"""N-vs-time performance plots: our OpenCL backend vs a reference JAX backend.

Each major op is swept over problem size N on two backends and plotted log-log
(one panel per op). To keep dispatch/launch jitter out of the numbers, every
benchmarked function applies its op CHAIN (=16) times as a data-dependent chain
inside ONE jitted call (optimization_barrier between links stops XLA fusing the
repeats); reported time is per op application, min over rounds. Because JAX's
backend is process-global (chosen at import from JAX_PLATFORMS), this script
runs each backend in its OWN subprocess:

  driver  (this file, no args)  ->  spawns one worker per backend, collects CSV,
                                    renders tools/bench_plot.png (+ .csv)
  worker  (--worker LABEL)      ->  backend fixed by env, benchmarks every
                                    (op, N), prints "op,N,ms" CSV to stdout

Reference backend is auto-detected: native CUDA jaxlib if installed (labelled
"JAX CUDA"), else JAX's CPU/XLA backend ("JAX CPU"). "Ours" is the pjrt-ocl
plugin on PJRT_OCL_DEVICE (default NVIDIA).

Usage:
    . ./env.sh && .venv/bin/python tools/plot_bench.py
    # options: --device NVIDIA  --engine mega|host  --out tools/bench_plot.png
"""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent

# --- op registry: name -> (build(N) -> (jitted_fn, args), sizes, xlabel) ------
# Sizes are chosen per op: element count for memory-bound ops, matrix side for
# matmul (O(N^3)). Each build() returns a jitted callable and its device args.
#
# Noise control: each jitted fn applies its op CHAIN times as a DATA-DEPENDENT
# chain (link i consumes link i-1's output), with lax.optimization_barrier
# between links so XLA cannot fuse/CSE the repeats into fewer kernels (our
# lowering aliases the barrier away — zero instructions). One timed call thus
# contains CHAIN real op executions but only ONE host dispatch, so per-call
# dispatch/launch jitter is amortized by 1/CHAIN in the reported per-op time.

CHAIN = 16


def _ops():
    import jax
    import jax.numpy as jnp
    from jax import lax

    rng = np.random.default_rng(0)
    f32 = np.float32

    def vec(n):
        return jnp.asarray(rng.standard_normal(n).astype(f32))

    def mat(n):
        return jnp.asarray(rng.standard_normal((n, n)).astype(f32))

    MEM = [1 << k for k in range(12, 25)]          # 4K .. 16M elements
    SQ = [128, 256, 384, 512, 768, 1024, 1536, 2048]

    def chain(step, *state):
        """jit a CHAIN-long data-dependent repetition of `step` over `state`.
        step maps the state tuple to a same-shape tuple; element 0 is the
        value being measured and is returned."""
        def f(*state):
            for _ in range(CHAIN):
                state = lax.optimization_barrier(step(*state))
            return state[0]
        return jax.jit(f), state

    def build_add(n):          # elementwise addition
        a, b = vec(n), vec(n)
        return chain(lambda x, y: (x + y, y), a, b)

    def build_mul(n):          # elementwise multiplication
        a, b = vec(n), vec(n)
        return chain(lambda x, y: (x * y, y), a, b)

    def build_matvec(n):       # matrix x vector: (n,n) @ (n,1) -> (n,1)
        a = mat(n) / np.sqrt(n)  # spectral radius ~1 keeps the chain finite
        v = jnp.asarray(rng.standard_normal((n, 1)).astype(f32))
        return chain(lambda y, x: (x @ y, x), v, a)

    def build_matmat(n):       # matrix x matrix
        a, b = mat(n), mat(n) / np.sqrt(n)
        return chain(lambda x, y: (x @ y, y), a, b)

    def build_gather(n):       # gather via dynamic_slice (runtime offset)
        a = vec(n)
        s = jnp.asarray(np.int32(n // 4))
        half = n // 2

        # Two traps (decisions.md §21) force this shape:
        # 1. Each link's offset must GENUINELY depend on the previous link's
        #    data — (y[0] * opaque_zero) is always 0 but fold-proof — or both
        #    backends DCE the first CHAIN-1 slices and the panel measures
        #    dispatch floor, not gathers.
        # 2. y must pass through a barrier BEFORE y[0], or XLA rewrites y[0]
        #    as a 1-element slice of x directly (slice-of-slice) and the big
        #    slices are dead again. The barrier forces y materialized.
        def f(x, k):
            z = lax.optimization_barrier(jnp.float32(0))
            y = lax.dynamic_slice(x, (k,), (half,))
            for _ in range(CHAIN - 1):
                y, k = lax.optimization_barrier((y, k))
                k = k + (y[0] * z).astype(jnp.int32)
                y = lax.dynamic_slice(x, (k,), (half,))
            return y
        return jax.jit(f), (a, s)

    def build_while(n):        # while loop: 32x elementwise update over vec(n)
        a = vec(n)
        return chain(lambda v: (lax.fori_loop(
            0, 32, lambda i, w: w * 1.5 + 1.0, v),), a)

    return {
        "elementwise add (a+b)":       (build_add, MEM, "N (elements)"),
        "elementwise mul (a*b)":       (build_mul, MEM, "N (elements)"),
        "matrix x vector (NxN . Nx1)": (build_matvec, SQ, "N (side)"),
        "matrix x matrix (NxN . NxN)": (build_matmat, SQ, "N (side)"),
        "gather (dynamic_slice)":      (build_gather, MEM, "N (elements)"),
        "while loop (32x a*1.5+1)":    (build_while, MEM, "N (elements)"),
    }


def _bench(fn, args, rounds=7, min_round_sec=0.05, max_iters=500):
    """Seconds per single op application (call time / CHAIN), min over rounds.

    The first call absorbs compile (for our backend, the lowering subprocess);
    a second timed call calibrates how many iterations a round needs to last
    >= min_round_sec, so short kernels aren't measured at timer resolution.
    min-of-rounds estimates the noise-free floor (timeit's rationale): every
    disturbance — scheduler preemption, clock ramp, GC — only ever adds time.
    """
    import jax

    jax.block_until_ready(fn(*args))                 # compile + first touch
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    once = time.perf_counter() - t0                  # rough steady-state cost
    iters = int(np.clip(min_round_sec / max(once, 1e-7), 1, max_iters))
    for _ in range(2):
        jax.block_until_ready(fn(*args))
    times = []
    for _ in range(rounds):
        t = time.perf_counter()
        for _ in range(iters):
            r = fn(*args)
        jax.block_until_ready(r)
        times.append((time.perf_counter() - t) / iters)
    return float(min(times)) / CHAIN


def run_worker(label: str) -> int:
    """Benchmark every (op, N) on the process's backend; print CSV to stdout."""
    ops = _ops()
    for name, (build, sizes, _x) in ops.items():
        for n in sizes:
            try:
                fn, args = build(n)
                sec = _bench(fn, args)
                print(f"{name},{n},{sec * 1e3:.6f}", flush=True)
            except Exception as e:  # unsupported op/size on this backend
                sys.stderr.write(f"skip {label} {name} N={n}: "
                                 f"{type(e).__name__}: {str(e)[:100]}\n")
                print(f"{name},{n},nan", flush=True)
    return 0


# --- driver ------------------------------------------------------------------


def detect_ref_backend() -> str:
    """'cuda' if a native CUDA jaxlib is present, else 'cpu'."""
    probe = ("import jax\n"
             "try:\n"
             "  jax.devices('cuda'); print('cuda')\n"
             "except Exception:\n"
             "  print('cpu')\n")
    env = dict(os.environ, JAX_PLATFORMS="cuda")
    try:
        out = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                             text=True, env=env, timeout=60)
        return "cuda" if "cuda" in out.stdout else "cpu"
    except Exception:
        return "cpu"


def worker_env(backend: str, device: str, engine: str | None) -> dict:
    env = dict(os.environ)
    env.setdefault("POCL_CACHE_DIR", str(REPO / "third_party" / "pocl-cache"))
    if backend == "ours":
        env["JAX_PLATFORMS"] = "opencl"
        env["PJRT_OCL_DEVICE"] = device
        # Only pin the plugin path if the dev-tree .so is actually there;
        # otherwise let the package's own search (env -> wheel -> dev tree)
        # find it. A wrong pinned path fails plugin discovery and every op
        # silently benches as NaN.
        so = REPO / "pjrt_plugin" / "build" / "libpjrt_ocl.so"
        if so.exists():
            env.setdefault("PJRT_OCL_PLUGIN_PATH", str(so))
        if engine:
            env["PJRT_OCL_ENGINE"] = engine
    else:
        env["JAX_PLATFORMS"] = backend  # 'cuda' or 'cpu'
    return env


def collect(label: str, env: dict) -> dict:
    """Run a worker subprocess, parse its CSV into {op: {N: ms}}."""
    proc = subprocess.run([sys.executable, str(pathlib.Path(__file__)),
                           "--worker", label], capture_output=True, text=True,
                          env=env, timeout=1800)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"worker {label} failed (rc={proc.returncode})")
    data: dict = {}
    for line in proc.stdout.splitlines():
        op, n, ms = line.rsplit(",", 2)
        data.setdefault(op, {})[int(n)] = float(ms)
    # A backend that failed to initialize benches every op as nan with the
    # real error hidden in stderr — surface it instead of plotting nothing.
    vals = [ms for per_op in data.values() for ms in per_op.values()]
    if vals and all(np.isnan(v) for v in vals):
        skips = proc.stderr.strip().splitlines()
        raise SystemExit(f"worker {label}: ALL results are nan — backend "
                         "broken, not 'unsupported ops'. First errors:\n  " +
                         "\n  ".join(skips[:5]))
    return data


def plot(ours: dict, ref: dict, ref_label: str, device: str, out: pathlib.Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ops = _ops()
    names = list(ops)
    ncol = 3
    nrow = (len(names) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow))
    axes = np.atleast_1d(axes).ravel()

    def series(d, op):
        xs = sorted(k for k, v in d.get(op, {}).items() if not np.isnan(v))
        return xs, [d[op][x] for x in xs]

    for ax, name in zip(axes, names):
        _b, _s, xlabel = ops[name]
        ox, oy = series(ours, name)
        rx, ry = series(ref, name)
        if ox:
            ax.plot(ox, oy, "o-", color="#d1495b", lw=2, ms=5,
                    label=f"ours (OpenCL/{device})")
        if rx:
            ax.plot(rx, ry, "s--", color="#30638e", lw=2, ms=5, label=ref_label)
        # speedup annotation at the largest common N (ratio = ours / ref)
        common = sorted(set(ox) & set(rx))
        if common:
            n = common[-1]
            ratio = ours[name][n] / ref[name][n]
            txt = (f"{ratio:.2g}x slower" if ratio >= 1
                   else f"{1 / ratio:.2g}x faster")
            ax.annotate(f"{txt}\n@ N={n:,}",
                        xy=(0.05, 0.92), xycoords="axes fraction",
                        va="top", fontsize=9,
                        bbox=dict(boxstyle="round", fc="#f7f7f7", ec="#ccc"))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("time / op (ms)")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(fontsize=8, loc="lower right")
    for ax in axes[len(names):]:
        ax.set_visible(False)

    fig.suptitle(f"pjrt-ocl: per-op N-vs-time — ours (OpenCL/{device}) vs "
                 f"{ref_label}", fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", metavar="LABEL",
                    help="internal: benchmark this process's backend")
    ap.add_argument("--device", default=os.environ.get("PJRT_OCL_DEVICE", "NVIDIA"))
    ap.add_argument("--engine", default=None, choices=[None, "mega", "host"],
                    help="force our engine (default: device auto)")
    ap.add_argument("--ref", default=None, choices=[None, "cuda", "cpu"],
                    help="reference backend (default: auto-detect cuda else cpu)")
    ap.add_argument("--out", default=str(REPO / "tools" / "bench_plot.png"))
    ap.add_argument("--replot", metavar="CSV",
                    help="skip benchmarking; re-render the plot from a CSV")
    ap.add_argument("--compare", nargs=2, metavar=("CSV_A", "CSV_B"),
                    help="skip benchmarking; plot CSV_A's ours_ms against "
                         "CSV_B's ours_ms (e.g. GPU vs CPU through the "
                         "plugin). --labels names the two sides.")
    ap.add_argument("--labels", nargs=2, default=["A", "B"],
                    help="with --compare: device labels for CSV_A / CSV_B")
    args = ap.parse_args()

    if args.worker:
        raise SystemExit(run_worker(args.worker))

    if args.compare:
        def ours_col(path):
            d: dict = {}
            with open(path) as f:
                next(f)
                for line in f:
                    op, n, om, _rm = line.rstrip("\n").rsplit(",", 3)
                    d.setdefault(op, {})[int(n)] = float(om)
            return d
        a, b = (ours_col(p) for p in args.compare)
        plot(a, b, f"ours (OpenCL/{args.labels[1]})", args.labels[0],
             pathlib.Path(args.out))
        return

    if args.replot:
        ours, refd, ref_label = {}, {}, "JAX CUDA (native)"
        with open(args.replot) as f:
            next(f)
            for line in f:
                op, n, om, rm = line.rstrip("\n").rsplit(",", 3)
                ours.setdefault(op, {})[int(n)] = float(om)
                refd.setdefault(op, {})[int(n)] = float(rm)
        plot(ours, refd, ref_label, args.device, pathlib.Path(args.out))
        return

    ref = args.ref or detect_ref_backend()
    ref_label = "JAX CUDA (native)" if ref == "cuda" else "JAX CPU (XLA)"
    print(f"reference backend: {ref_label}; ours: OpenCL/{args.device}"
          f"{' engine=' + args.engine if args.engine else ''}")

    print("benchmarking ours ...", flush=True)
    ours = collect("ours", worker_env("ours", args.device, args.engine))
    print("benchmarking reference ...", flush=True)
    refd = collect("ref", worker_env(ref, args.device, None))

    out = pathlib.Path(args.out)
    # dump raw CSV next to the PNG for reproducibility
    csv = out.with_suffix(".csv")
    with open(csv, "w") as f:
        f.write("op,N,ours_ms,ref_ms\n")
        for op in _ops():
            for n in sorted(set(ours.get(op, {})) | set(refd.get(op, {}))):
                f.write(f"{op},{n},{ours.get(op, {}).get(n, float('nan'))},"
                        f"{refd.get(op, {}).get(n, float('nan'))}\n")
    print(f"wrote {csv}")
    plot(ours, refd, ref_label, args.device, out)


if __name__ == "__main__":
    main()
