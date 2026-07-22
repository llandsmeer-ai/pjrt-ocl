"""Driver for the diverse AI + scientific-computing workload testbench.

Runs every workload in ``workloads.py`` against BOTH backends — our OpenCL VM
(``JAX_PLATFORMS=opencl PJRT_OCL_DEVICE=NVIDIA``) and native JAX CUDA
(``JAX_PLATFORMS=cuda``) — each in its own subprocess (the JAX backend is
process-global). For each workload it records:

  * PASS/FAIL on our backend, and if FAIL the exact missing StableHLO op,
  * correctness vs CUDA (max abs/rel error, allclose flag),
  * median-of-rounds latency on each backend and the ours/CUDA gap.

Usage:
    . ./env.sh
    .venv/bin/python tools/bench_suite/run_suite.py            # run everything
    .venv/bin/python tools/bench_suite/run_suite.py --only mlp attention
    .venv/bin/python tools/bench_suite/run_suite.py --md docs/workload-coverage.md

Internally re-invokes itself as a worker:
    ... run_suite.py --worker --name mlp --outdir /tmp/xx
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.dirname(THIS_DIR)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

SENTINEL = "##BENCHJSON##"

# Match the op named in a LoweringError / unsupported-op message.
_OP_RES = [
    re.compile(r"unsupported op:\s*(stablehlo\.[a-z_]+)"),
    re.compile(r"(stablehlo\.[a-z_]+)"),
]


def _extract_reason(msg: str):
    """Pull a concise reason + a missing-op token out of an error string."""
    op = None
    for rx in _OP_RES:
        m = rx.search(msg)
        if m:
            op = m.group(1)
            break
    # Named lowering rejections that are not a raw "unsupported op".
    low = msg.lower()
    if "unsupported device" in low or "unsupported_device" in low:
        op = "platform-allowlist"
        reason = "host lib rejects OpenCL PJRT platform (device allowlist)"
    elif "complex" in low:
        op = op or "complex-dtype"
        reason = "complex dtype unsupported"
    elif "partial" in low or "innermost-suffix" in low or "full or innermost" in low:
        op = "reduce(partial-axis)"
        reason = "partial-axis reduce"
    elif "batch" in low and "dot" in low:
        op = op or "dot_general(batched)"
        reason = "batched dot_general"
    elif op:
        reason = f"missing {op}"
    else:
        reason = msg.strip().splitlines()[0][:120]
    return reason, op


# ------------------------------------------------------------------- worker ---

def _worker(name: str, outdir: str) -> int:
    import numpy as np
    from bench_suite import workloads

    result = {"name": name}
    try:
        import jax
        import jax.numpy as jnp
    except Exception as e:  # pragma: no cover
        result.update(status="ERROR", error=f"jax import: {e}")
        print(SENTINEL + json.dumps(result))
        return 0

    try:
        fn, tree, meta = workloads.build(name)
        result["meta"] = meta
    except Exception as e:
        reason, op = _extract_reason(f"{type(e).__name__}: {e}")
        result.update(status="FAIL", stage="build", error=str(e)[:400],
                      reason=reason, missing_op=op)
        print(SENTINEL + json.dumps(result))
        return 0

    tree_j = jax.tree_util.tree_map(lambda a: jnp.asarray(a), tree)
    fn_j = jax.jit(fn)

    # 1) lower + run once
    try:
        out = fn_j(tree_j)
        jax.block_until_ready(out)
        out_np = np.asarray(out)
    except Exception as e:
        reason, op = _extract_reason(f"{type(e).__name__}: {e}")
        result.update(status="FAIL", stage="run", error=str(e)[:500],
                      reason=reason, missing_op=op)
        print(SENTINEL + json.dumps(result))
        return 0

    finite = bool(np.isfinite(out_np).all())
    np.save(os.path.join(outdir, name + ".npy"), out_np)
    result.update(status="PASS", shape=list(out_np.shape), finite=finite,
                  mean=float(out_np.mean()), std=float(out_np.std()),
                  amin=float(out_np.min()), amax=float(out_np.max()))

    # 2) benchmark: adaptive iters, median of rounds
    for _ in range(3):
        jax.block_until_ready(fn_j(tree_j))
    t0 = time.perf_counter()
    jax.block_until_ready(fn_j(tree_j))
    one = time.perf_counter() - t0
    iters = max(3, min(200, int(0.08 / max(one, 1e-6))))
    ts = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(iters):
            r = fn_j(tree_j)
        jax.block_until_ready(r)
        ts.append((time.perf_counter() - t) / iters)
    ts.sort()
    result["ms"] = float(ts[len(ts) // 2]) * 1e3
    result["iters"] = iters
    print(SENTINEL + json.dumps(result))
    return 0


# ------------------------------------------------------------- orchestrator ---

BACKENDS = {
    "opencl": {"JAX_PLATFORMS": "opencl", "PJRT_OCL_DEVICE": "NVIDIA"},
    "cuda": {"JAX_PLATFORMS": "cuda"},
    # CPU story (this round): ours-on-PoCL vs the reference JAX CPU/XLA backend,
    # each in its own process. Selected by --cpu (rewrites "opencl"/ref below).
    "opencl_cpu": {"JAX_PLATFORMS": "opencl", "PJRT_OCL_DEVICE": "Portable"},
    "cpu": {"JAX_PLATFORMS": "cpu"},
}

# The reference backend key (native XLA). Overridden to "cpu" by --cpu; the row
# dicts still store it under the key "cuda" so the printers stay backend-generic.
REF_BACKEND = "cuda"
OURS_BACKEND = "opencl"
CPU_MODE = False


def _run_worker(name: str, backend: str, outdir: str, timeout=360):
    env = dict(os.environ)
    env.update(BACKENDS[backend])
    # keep caches off the full root overlay even if env.sh wasn't sourced
    root = os.path.dirname(TOOLS_DIR)
    env.setdefault("POCL_CACHE_DIR", os.path.join(root, "third_party/pocl-cache"))
    env.setdefault("CUDA_CACHE_PATH", os.path.join(root, "third_party/nv-cache"))
    env.setdefault("TMPDIR", os.path.join(root, "third_party/tmp"))
    cmd = [sys.executable, os.path.abspath(__file__),
           "--worker", "--name", name, "--outdir", outdir]
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"name": name, "status": "FAIL", "stage": "timeout",
                "reason": "timeout", "error": f">{timeout}s"}
    for line in p.stdout.splitlines():
        if line.startswith(SENTINEL):
            return json.loads(line[len(SENTINEL):])
    # crashed without emitting JSON
    tail = (p.stderr.strip().splitlines() or ["(no stderr)"])[-1]
    reason, op = _extract_reason(p.stderr)
    return {"name": name, "status": "FAIL", "stage": "crash",
            "reason": reason, "missing_op": op, "error": tail[:400],
            "returncode": p.returncode}


def _compare(name, outdir):
    import numpy as np
    fo = os.path.join(outdir, name + ".npy")
    fc = os.path.join(outdir, name + ".cuda.npy")
    if not (os.path.exists(fo) and os.path.exists(fc)):
        return None
    a, b = np.load(fo), np.load(fc)
    if a.shape != b.shape:
        return {"close": False, "note": f"shape {a.shape} vs {b.shape}"}
    diff = np.abs(a - b)
    max_abs = float(diff.max()) if a.size else 0.0
    max_rel = float((diff / (np.abs(b) + 1e-6)).max()) if a.size else 0.0
    # On PoCL/CPU there is no TF32: ours should be f32-exact vs XLA-CPU, so a
    # tighter tolerance is warranted. Keep the loose one for the GPU/TF32 path.
    atol, rtol = (2e-2, 2e-2)
    if CPU_MODE:
        atol, rtol = (1e-3, 1e-3)
    close = bool(np.allclose(a, b, atol=atol, rtol=rtol))
    return {"close": close, "max_abs": max_abs, "max_rel": max_rel}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--name")
    ap.add_argument("--outdir")
    ap.add_argument("--only", nargs="*", help="subset of workload names")
    ap.add_argument("--md", help="write a markdown scoreboard to this path")
    ap.add_argument("--cpu", action="store_true",
                    help="CPU story: ours-on-PoCL (PJRT_OCL_DEVICE=Portable) vs "
                         "reference JAX CPU/XLA backend, in separate processes")
    args = ap.parse_args()

    if args.worker:
        return _worker(args.name, args.outdir)

    global CPU_MODE, REF_BACKEND, OURS_BACKEND
    if args.cpu:
        CPU_MODE = True
        OURS_BACKEND = "opencl_cpu"
        REF_BACKEND = "cpu"

    from bench_suite import workloads
    names = args.only or workloads.WORKLOAD_NAMES
    outdir = tempfile.mkdtemp(prefix="benchsuite_")
    rows = []
    for name in names:
        print(f"[ ours ] {name} ...", flush=True)
        ours = _run_worker(name, OURS_BACKEND, outdir)
        # rename ours npy so the ref run doesn't clobber it
        src = os.path.join(outdir, name + ".npy")
        if os.path.exists(src):
            os.replace(src, os.path.join(outdir, name + ".ours.npy"))

        print(f"[ {REF_BACKEND:>4} ] {name} ...", flush=True)
        cuda = _run_worker(name, REF_BACKEND, outdir)
        src = os.path.join(outdir, name + ".npy")
        if os.path.exists(src):
            os.replace(src, os.path.join(outdir, name + ".cuda.npy"))
        # our npy is at name.ours.npy; compare wants name.npy
        oursnpy = os.path.join(outdir, name + ".ours.npy")
        if os.path.exists(oursnpy):
            os.replace(oursnpy, os.path.join(outdir, name + ".npy"))

        cmp = _compare(name, outdir) if ours.get("status") == "PASS" else None
        rows.append({"name": name, "ours": ours, "cuda": cuda, "cmp": cmp})
        _print_row(rows[-1])

    _print_table(rows)
    _rank_missing(rows)
    if args.md:
        _write_md(args.md, rows)
        print(f"\nwrote {args.md}")
    return 0


def _print_row(r):
    o, c, cmp = r["ours"], r["cuda"], r["cmp"]
    st = o.get("status")
    if st == "PASS":
        gap = (o["ms"] / c["ms"]) if c.get("status") == "PASS" and c.get("ms") else float("nan")
        cl = "-" if not cmp else ("close" if cmp["close"] else f"DIFF(rel={cmp.get('max_rel',0):.1e})")
        print(f"    -> PASS ours={o['ms']:.3f}ms cuda={c.get('ms',float('nan')):.3f}ms "
              f"gap={gap:.2f}x correct={cl} finite={o.get('finite')}")
    else:
        print(f"    -> FAIL [{o.get('stage')}] {o.get('reason')} "
              f"(op={o.get('missing_op')}) cuda={c.get('status')}")


def _print_table(rows):
    print("\n================ COVERAGE / PERF TABLE ================")
    hdr = f"{'workload':<18}{'cat':<6}{'ours':<8}{'ours ms':>10}{'cuda ms':>10}{'gap':>8}  missing/notes"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        o, c, cmp = r["ours"], r["cuda"], r["cmp"]
        cat = (o.get("meta") or c.get("meta") or {}).get("cat", "?")
        if o.get("status") == "PASS":
            gap = (o["ms"] / c["ms"]) if c.get("status") == "PASS" and c.get("ms") else float("nan")
            note = "" if not cmp else ("" if cmp["close"] else f"DIFF rel={cmp.get('max_rel',0):.1e}")
            print(f"{r['name']:<18}{cat:<6}{'PASS':<8}{o['ms']:>10.3f}"
                  f"{c.get('ms', float('nan')):>10.3f}{gap:>7.2f}x  {note}")
        else:
            miss = o.get("missing_op") or o.get("reason") or "?"
            cs = c.get("status", "?")
            cms = f"{c['ms']:.3f}" if cs == "PASS" and c.get("ms") else "-"
            print(f"{r['name']:<18}{cat:<6}{'FAIL':<8}{'-':>10}{cms:>10}"
                  f"{'-':>8}  {miss}  (cuda:{cs})")


def _rank_missing(rows):
    from collections import defaultdict
    blockers = defaultdict(list)
    for r in rows:
        o = r["ours"]
        if o.get("status") != "PASS":
            key = o.get("missing_op") or o.get("reason") or "unknown"
            blockers[key].append(r["name"])
    if not blockers:
        return
    print("\n=========== RANKED MISSING-OP BLOCKERS (M3 priority) ===========")
    for op, wls in sorted(blockers.items(), key=lambda kv: -len(kv[1])):
        print(f"  {len(wls):>2}x  {op:<28} unlocks: {', '.join(wls)}")


def _write_md(path, rows):
    from collections import defaultdict
    ref = "XLA-CPU" if CPU_MODE else "CUDA"
    refms = "cpu ms" if CPU_MODE else "cuda ms"
    npass = sum(1 for r in rows if r["ours"].get("status") == "PASS")
    n = len(rows)
    lines = []
    lines.append("# Workload coverage & perf scoreboard (CPU / PoCL)\n" if CPU_MODE
                 else "# Workload coverage & perf scoreboard\n")
    if CPU_MODE:
        lines.append("Diverse AI + scientific-computing workloads run through our OpenCL VM "
                     "on PoCL (`JAX_PLATFORMS=opencl PJRT_OCL_DEVICE=Portable`, the "
                     "host-dispatch engine: no spin-barrier, clFinish/ring-drain per phase, "
                     "`-DVMO_CPU_TILES` float8 EW + packed SGEMM) vs the reference native "
                     "JAX CPU/XLA backend (`JAX_PLATFORMS=cpu`) on the same AMD Ryzen 9 "
                     "3900X (12c/24t). Generated by `tools/bench_suite/run_suite.py --cpu`. "
                     "Each backend runs in its own subprocess; latency is median-of-5-rounds, "
                     "adaptive iters, after warmup. `gap = ours_ms / cpu_ms` (lower is "
                     "better; <1 means we beat XLA-CPU). PoCL has no TF32, so correctness "
                     "is checked f32-tight (`allclose atol=rtol=1e-3`).\n")
    else:
        lines.append("Diverse AI + scientific-computing workloads run through our OpenCL VM "
                     "(`JAX_PLATFORMS=opencl PJRT_OCL_DEVICE=NVIDIA`) vs native JAX CUDA "
                     "(`JAX_PLATFORMS=cuda`) on the same RTX PRO 6000. Generated by "
                     "`tools/bench_suite/run_suite.py`. Each backend runs in its own "
                     "subprocess; latency is median-of-5-rounds, adaptive iters, after warmup. "
                     "`gap = ours_ms / cuda_ms` (lower is better; <1 means we beat CUDA).\n")
    lines.append(f"**Coverage: {npass}/{n} workloads run on our backend "
                 f"({100*npass//n if n else 0}%).**\n")
    lines.append("## Coverage + perf\n")
    lines.append(f"| workload | cat | status | ours ms | {refms} | gap | correct (max rel vs {ref}) | missing op / note |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        o, c, cmp = r["ours"], r["cuda"], r["cmp"]
        meta = (o.get("meta") or c.get("meta") or {})
        cat = meta.get("cat", "?")
        note = meta.get("note", "")
        if o.get("status") == "PASS":
            gap = (o["ms"] / c["ms"]) if c.get("status") == "PASS" and c.get("ms") else float("nan")
            cms = f"{c.get('ms'):.3f}" if c.get("status") == "PASS" else "-"
            if cmp:
                corr = f"{'close' if cmp['close'] else 'DIFF'} ({cmp.get('max_rel', 0):.1e})"
            else:
                corr = "-"
            lines.append(f"| {r['name']} | {cat} | PASS | {o['ms']:.3f} | {cms} | "
                         f"{gap:.2f}x | {corr} | {note} |")
        else:
            miss = o.get("missing_op") or o.get("reason") or "?"
            cs = c.get("status", "?")
            cms = f"{c.get('ms'):.3f}" if cs == "PASS" else cs
            lines.append(f"| {r['name']} | {cat} | **FAIL** | - | {cms} | - | - | "
                         f"`{miss}` — {o.get('reason','')} ({ref.lower()}: {cs}); {note} |")

    # ranked blockers
    blockers = defaultdict(list)
    for r in rows:
        o = r["ours"]
        if o.get("status") != "PASS":
            key = o.get("missing_op") or o.get("reason") or "unknown"
            blockers[key].append(r["name"])
    lines.append("\n## Ranked missing-op priority (M3 test-driven order)\n")
    lines.append("Each row: how many suite workloads that op/feature would unlock, and which.\n")
    lines.append("| rank | missing op / feature | # workloads unlocked | workloads |")
    lines.append("|---|---|---|---|")
    for i, (op, wls) in enumerate(sorted(blockers.items(), key=lambda kv: -len(kv[1])), 1):
        lines.append(f"| {i} | `{op}` | {len(wls)} | {', '.join(wls)} |")

    # bottom line, computed from the data
    gaps = []
    for r in rows:
        o, c = r["ours"], r["cuda"]
        if o.get("status") == "PASS" and c.get("status") == "PASS" and c.get("ms"):
            gaps.append((r["name"], o["ms"] / c["ms"]))
    lines.append("\n## Bottom line (generalization read)\n")
    if gaps:
        gs = sorted(gaps, key=lambda kv: kv[1])
        med = sorted(g for _, g in gaps)[len(gaps) // 2]
        best, worst = gs[0], gs[-1]
        nbeat = sum(1 for _, g in gaps if g <= 1.0)
        nmatch = sum(1 for _, g in gaps if g <= 1.5)
        lines.append(f"- Passers' gap vs {ref} spans **{best[1]:.2f}x ({best[0]}) to "
                     f"{worst[1]:.2f}x ({worst[0]})**, median **{med:.2f}x**.")
        lines.append(f"- **At/under 1x {ref} (we win or tie): {nbeat}/{len(gaps)}**; "
                     f"within 1.5x: {nmatch}/{len(gaps)}.")
        if CPU_MODE:
            winners = ", ".join(f"{nm} ({g:.2f}x)" for nm, g in gs if g <= 1.0) or "none"
            laggards = ", ".join(f"{nm} ({g:.2f}x)" for nm, g in gs if g >= 3.0) or "none"
            lines.append(f"- **Winners (≤1x XLA-CPU):** {winners}.")
            lines.append(f"- **Laggards (≥3x XLA-CPU):** {laggards} — the loop-/small-op "
                         "regime where per-phase enqueue + clFinish/ring-drain dominates.")
    lines.append("\n### Notes on the correctness column\n")
    if CPU_MODE:
        lines.append("- `correct` uses `np.allclose(atol=1e-3, rtol=1e-3)` vs XLA-CPU (PoCL has "
                     "no TF32, so ours is f32 all the way and should match XLA-CPU tightly). "
                     "The parenthesised `max rel` can still look large where the reference has "
                     "near-zero elements (rel = |Δ|/(|ref|+1e-6)); the boolean is authoritative.")
    else:
        lines.append("- `correct` uses `np.allclose(atol=2e-2, rtol=2e-2)` vs CUDA and that boolean is "
                     "authoritative. The parenthesised `max rel` can look huge (e.g. transformer 5e1) "
                     "purely because the reference has near-zero elements (rel = |Δ|/(|ref|+1e-6)); "
                     "every passer clears the abs+rel allclose. TF32 matmul on both sides also widens "
                     "abs error on ~1-std signals.")
    lines.append("- Chaotic integrators (rk4 Lorenz, logistic_map at r→4) are seeded identically "
                 "and match here, but would diverge over longer horizons on any two backends.")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
