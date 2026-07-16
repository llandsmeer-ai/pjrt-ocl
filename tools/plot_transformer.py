"""End-to-end transformer benchmark plot: our OpenCL backend vs native JAX CUDA.

Sweeps the GPT-style forward pass (tools/bench_transformer.py) over model size on
both backends and renders ONE figure (docs/bench_transformer.png + .csv):

  left panel  — ms/iter per config, ours vs reference, gap annotated
  right panel — sustained throughput (TFLOP/s), showing ours climb into the
                compute-bound regime as the model grows

Backend is process-global (chosen at import from JAX_PLATFORMS), so each
(config, backend) point is measured in its own `bench_transformer.py`
subprocess and its "… ms/iter … GFLOP/s" line is parsed. Reference is native
CUDA jaxlib if present, else JAX CPU.

Usage:
    . ./env.sh && .venv/bin/python tools/plot_transformer.py
    # options: --device NVIDIA  --iters 30  --out docs/bench_transformer.png
    #          --configs tiny,small,base,large_l1,large   --replot CSV
"""
from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
BENCH = REPO / "tools" / "bench_transformer.py"
_LINE = re.compile(r"([0-9.]+)\s*ms/iter\s+([0-9.]+)\s*GFLOP/s")


def _flops(cfg):
    import importlib.util
    spec = importlib.util.spec_from_file_location("bt", BENCH)
    bt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bt)
    return bt.flops(bt.CONFIGS[cfg]), bt.CONFIGS[cfg]


def _measure(cfg: str, backend: str, device: str, iters: int) -> tuple[float, float]:
    """(ms/iter, GFLOP/s) for one config on one backend, via a subprocess."""
    env = dict(os.environ)
    env.setdefault("POCL_CACHE_DIR", str(REPO / "third_party" / "pocl-cache"))
    if backend == "ours":
        env["JAX_PLATFORMS"] = "opencl"
        env["PJRT_OCL_DEVICE"] = device
        so = REPO / "pjrt_plugin" / "build" / "libpjrt_ocl.so"
        if so.exists():
            env.setdefault("PJRT_OCL_PLUGIN_PATH", str(so))
    else:
        env["JAX_PLATFORMS"] = backend            # 'cuda' or 'cpu'
    proc = subprocess.run([sys.executable, str(BENCH), "--config", cfg,
                           "--iters", str(iters)], capture_output=True,
                          text=True, env=env, timeout=1200)
    if proc.returncode != 0:
        sys.stderr.write(f"[{backend} {cfg}] rc={proc.returncode}\n{proc.stderr[-800:]}\n")
        return float("nan"), float("nan")
    m = _LINE.search(proc.stdout)
    if not m:
        sys.stderr.write(f"[{backend} {cfg}] no timing line:\n{proc.stdout}\n")
        return float("nan"), float("nan")
    return float(m.group(1)), float(m.group(2))


def detect_ref() -> str:
    probe = ("import jax\ntry:\n jax.devices('cuda'); print('cuda')\n"
             "except Exception:\n print('cpu')\n")
    try:
        out = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                             text=True, env=dict(os.environ, JAX_PLATFORMS="cuda"),
                             timeout=60)
        return "cuda" if "cuda" in out.stdout else "cpu"
    except Exception:
        return "cpu"


def plot(rows, ref_label, device, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in rows if not np.isnan(r["ours_ms"])]
    labels = [r["cfg"] for r in rows]
    x = np.arange(len(rows))
    w = 0.38
    OURS, REF = "#d1495b", "#30638e"
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))

    # left: ms/iter (log)
    ours_ms = [r["ours_ms"] for r in rows]
    ref_ms = [r["ref_ms"] for r in rows]
    axL.bar(x - w / 2, ours_ms, w, color=OURS, label=f"ours (OpenCL/{device})")
    axL.bar(x + w / 2, ref_ms, w, color=REF, label=ref_label)
    axL.set_yscale("log")
    for i, r in enumerate(rows):
        if not np.isnan(r["ref_ms"]):
            axL.annotate(f"{r['ours_ms'] / r['ref_ms']:.1f}×",
                         xy=(i - w / 2, r["ours_ms"]), ha="center", va="bottom",
                         fontsize=9, fontweight="bold", color=OURS)
    axL.set_ylabel("time / iter (ms, log)")
    axL.set_title("Forward-pass latency vs. native")

    # right: sustained throughput (TFLOP/s)
    ours_tf = [r["ours_gf"] / 1000 for r in rows]
    ref_tf = [r["ref_gf"] / 1000 for r in rows]
    axR.bar(x - w / 2, ours_tf, w, color=OURS, label=f"ours (OpenCL/{device})")
    axR.bar(x + w / 2, ref_tf, w, color=REF, label=ref_label)
    for i, r in enumerate(rows):
        axR.annotate(f"{r['ours_gf'] / 1000:.1f}", xy=(i - w / 2, r["ours_gf"] / 1000),
                     ha="center", va="bottom", fontsize=9, color=OURS)
    axR.set_ylabel("sustained throughput (TFLOP/s)")
    axR.set_title("Throughput — ours climbs as work gets compute-bound")

    for ax in (axL, axR):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        ax.grid(True, axis="y", ls=":", alpha=0.4)
        ax.legend(fontsize=9, loc="upper left")

    fig.suptitle(f"pjrt-ocl: GPT-style transformer forward pass — ours "
                 f"(OpenCL/{device}) vs {ref_label}", fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=os.environ.get("PJRT_OCL_DEVICE", "NVIDIA"))
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--configs", default="tiny,small,base,large_l1,large")
    ap.add_argument("--ref", default=None, choices=[None, "cuda", "cpu"])
    ap.add_argument("--out", default=str(REPO / "docs" / "bench_transformer.png"))
    ap.add_argument("--replot", metavar="CSV", help="re-render from a CSV, skip benchmarking")
    args = ap.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    ref = args.ref or detect_ref()
    ref_label = "JAX CUDA (native)" if ref == "cuda" else "JAX CPU (XLA)"
    out = pathlib.Path(args.out)
    csv = out.with_suffix(".csv")

    if args.replot:
        rows = []
        with open(args.replot) as f:
            next(f)
            for line in f:
                cfg, fl, om, og, rm, rg = line.rstrip("\n").split(",")
                rows.append(dict(cfg=cfg, flops=float(fl), ours_ms=float(om),
                                 ours_gf=float(og), ref_ms=float(rm), ref_gf=float(rg)))
        plot(rows, ref_label, args.device, out)
        return

    print(f"reference: {ref_label}; ours: OpenCL/{args.device}; iters={args.iters}")
    rows = []
    for cfg in configs:
        fl, _ = _flops(cfg)
        oms, ogf = _measure(cfg, "ours", args.device, args.iters)
        rms, rgf = _measure(cfg, ref, args.device, args.iters)
        gap = oms / rms if rms else float("nan")
        print(f"  {cfg:9s} ours {oms:8.3f} ms {ogf:8.1f} GF/s | "
              f"{ref} {rms:7.3f} ms {rgf:9.1f} GF/s | gap {gap:.1f}×")
        rows.append(dict(cfg=cfg, flops=fl, ours_ms=oms, ours_gf=ogf,
                         ref_ms=rms, ref_gf=rgf))

    with open(csv, "w") as f:
        f.write("config,flops,ours_ms,ours_gflops,ref_ms,ref_gflops\n")
        for r in rows:
            f.write(f"{r['cfg']},{r['flops']},{r['ours_ms']},{r['ours_gf']},"
                    f"{r['ref_ms']},{r['ref_gf']}\n")
    print(f"wrote {csv}")
    plot(rows, ref_label, args.device, out)


if __name__ == "__main__":
    main()
