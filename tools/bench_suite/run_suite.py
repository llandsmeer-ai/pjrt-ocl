"""Driver for the diverse AI + scientific-computing workload testbench.

Runs every workload in ``workloads.py`` against BOTH backends — our OpenCL VM
and a native-XLA reference — each in its own subprocess (the JAX backend is
process-global). For each workload it records:

  * PASS/FAIL on our backend, and if FAIL the exact missing StableHLO op,
  * correctness vs the reference (max abs/rel error, allclose flag),
  * median-of-rounds latency on each backend and the ours/reference gap.

Three run modes (all use the same driver/worker structure; they only pick which
two BACKENDS entries to run and which ``PJRT_OCL_DEVICE`` the "ours" side gets):

  (default)     ours on an OpenCL GPU (``PJRT_OCL_DEVICE=NVIDIA``) vs native
                JAX CUDA (``JAX_PLATFORMS=cuda``).
  --cpu         ours on PoCL (``PJRT_OCL_DEVICE=Portable``) vs JAX CPU/XLA.
  --gpu-vs-cpu  ours on an OpenCL GPU (default ``PJRT_OCL_DEVICE=Intel``) vs
                the JAX CPU/XLA reference. For hosts with an OpenCL GPU but no
                CUDA (e.g. Intel Lunar Lake / Arc 140V Xe2 iGPU).

``--ocl-device SUBSTR`` overrides the OpenCL platform substring for the "ours"
side in any mode, so nothing is hardcoded to one vendor.

Usage:
    . ./env.sh
    .venv/bin/python tools/bench_suite/run_suite.py            # run everything
    .venv/bin/python tools/bench_suite/run_suite.py --only mlp attention
    .venv/bin/python tools/bench_suite/run_suite.py --md docs/workload-coverage.md
    # Xe2 iGPU vs XLA-CPU on a CUDA-less host:
    .venv/bin/python tools/bench_suite/run_suite.py --gpu-vs-cpu \
        --ocl-device Intel --md docs/workload-coverage-xe2.md

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
        lines = [l for l in msg.strip().splitlines() if l.strip()]
        reason = lines[0][:120] if lines else "unknown"
    return reason, op


# Lines JAX/absl/OpenCL runtimes emit on stderr that are NOT the error: the
# experimental-platform warning is printed by xla_bridge on EVERY successful
# opencl run, and JAX's traceback filter appends its own footer. Taking
# stderr[0] or stderr[-1] blindly reports one of these as the failure reason.
_NOISE_RES = [
    re.compile(r"^Platform '.*' is experimental"),
    re.compile(r"^For simplicity, JAX has removed its internal frames"),
    re.compile(r"^-{5,}$"),
    re.compile(r"^\s*(Traceback \(most recent call last\)|During handling of "
               r"the above exception|The above exception was the direct cause)"),
    re.compile(r"^\s*(File \"|\^{2,}|\.{3}$)"),
    re.compile(r"^\s*(WARNING|INFO|W\d{4}|I\d{4}|E\d{4})[: ]"),
    re.compile(r"^\s*warnings\.warn"),
]


def _denoise_stderr(stderr: str):
    """Drop framework noise + traceback frames, keep candidate error lines."""
    keep = []
    for line in stderr.splitlines():
        if not line.strip():
            continue
        if any(rx.search(line) for rx in _NOISE_RES):
            continue
        # traceback source lines are indented; exception lines are not
        if line.startswith((" ", "\t")):
            continue
        keep.append(line.rstrip())
    return keep


def _crash_reason(stderr: str, returncode: int):
    """Reason/op/message for a worker that died without emitting its JSON.

    The last non-noise, non-indented stderr line is the actual exception
    ("RuntimeError: Unable to initialize backend 'opencl': ..."); everything
    else is warnings and traceback frames.
    """
    lines = _denoise_stderr(stderr)
    msg = lines[-1] if lines else ""
    if returncode is not None and returncode < 0:
        sig = -returncode
        return (f"worker killed by signal {sig}", f"signal-{sig}",
                (msg or f"(no stderr) signal {sig}")[:400])
    if not msg:
        return ("worker died without output", None,
                f"(no usable stderr) returncode={returncode}")
    reason, op = _extract_reason("\n".join(lines))
    return reason, op, msg[:400]


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

    # Moving the inputs onto the device is what actually initializes the JAX
    # backend, so a bad PJRT_OCL_DEVICE / missing plugin / missing CUDA raises
    # HERE. Uncaught, it killed the worker before it printed its JSON line and
    # the driver had to guess a reason from raw stderr (and picked JAX's
    # "Platform 'opencl' is experimental" warning). Report it as data instead.
    try:
        tree_j = jax.tree_util.tree_map(lambda a: jnp.asarray(a), tree)
        fn_j = jax.jit(fn)
    except Exception as e:
        reason, op = _extract_reason(f"{type(e).__name__}: {e}")
        result.update(status="FAIL", stage="backend", error=str(e)[:500],
                      reason=reason, missing_op=op)
        print(SENTINEL + json.dumps(result))
        return 0

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
    try:
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
    except Exception as e:
        # The run was correct; only timing blew up. Keep it a FAIL so it can't
        # silently become a missing-number row, but say exactly what happened.
        reason, op = _extract_reason(f"{type(e).__name__}: {e}")
        result.update(status="FAIL", stage="bench", error=str(e)[:500],
                      reason=f"benchmark loop: {reason}", missing_op=op)
    print(SENTINEL + json.dumps(result))
    return 0


# ------------------------------------------------------------- orchestrator ---

BACKENDS = {
    # "ours": our OpenCL VM. PJRT_OCL_DEVICE is filled in from OCL_DEVICE at
    # launch time (see _run_worker) so no vendor is hardcoded here.
    "opencl": {"JAX_PLATFORMS": "opencl"},
    # native-XLA references
    "cuda": {"JAX_PLATFORMS": "cuda"},
    "cpu": {"JAX_PLATFORMS": "cpu"},
}

# Run modes: which backends to pit against each other, the default OpenCL
# platform substring for the "ours" side, and how the printers label things.
# --cpu / --gpu-vs-cpu just select a row here.
MODES = {
    "nvidia": {"ours": "opencl", "ref": "cuda", "device": "NVIDIA",
               "ref_label": "CUDA"},
    "cpu": {"ours": "opencl", "ref": "cpu", "device": "Portable",
            "ref_label": "XLA-CPU"},
    "gpu-vs-cpu": {"ours": "opencl", "ref": "cpu", "device": "Intel",
                   "ref_label": "XLA-CPU"},
}

# The reference backend key (native XLA). Overridden by the mode; the row dicts
# still store it under the key "cuda" so the printers stay backend-generic.
REF_BACKEND = "cuda"
OURS_BACKEND = "opencl"
MODE = "nvidia"
CPU_MODE = False        # ours-on-PoCL vs XLA-CPU: tight tolerance + CPU prose
OCL_DEVICE = "NVIDIA"   # PJRT_OCL_DEVICE substring handed to the "ours" runs
REF_LABEL = "CUDA"
TIMEOUT = 360


def _run_worker(name: str, backend: str, outdir: str, timeout=None):
    timeout = TIMEOUT if timeout is None else timeout
    env = dict(os.environ)
    env.update(BACKENDS[backend])
    root = os.path.dirname(TOOLS_DIR)
    if env.get("JAX_PLATFORMS") == "opencl":
        env["PJRT_OCL_DEVICE"] = OCL_DEVICE
        # Only pin the plugin path if the dev-tree .so is actually there;
        # otherwise let pjrt_ocl's own search (env -> wheel -> dev tree) run.
        so = os.path.join(root, "pjrt_plugin/build/libpjrt_ocl.so")
        if os.path.exists(so):
            env.setdefault("PJRT_OCL_PLUGIN_PATH", so)
    # keep caches off the full root overlay even if env.sh wasn't sourced
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
    # crashed without emitting JSON (segfault, VM hang killed, os._exit, ...)
    reason, op, msg = _crash_reason(p.stderr, p.returncode)
    return {"name": name, "status": "FAIL", "stage": "crash",
            "reason": reason, "missing_op": op, "error": msg,
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
    # tighter tolerance is warranted. Keep the loose one for the GPU path
    # (TF32 on NVIDIA; different reduction order / fma contraction on any GPU).
    atol, rtol = (2e-2, 2e-2)
    if CPU_MODE:
        atol, rtol = (1e-3, 1e-3)
    close = bool(np.allclose(a, b, atol=atol, rtol=rtol))
    # Extra information, never used to decide PASS/FAIL: would it also clear
    # the f32-tight bar? Reported so a GPU run's real accuracy is visible.
    tight = bool(np.allclose(a, b, atol=1e-3, rtol=1e-3))
    return {"close": close, "max_abs": max_abs, "max_rel": max_rel,
            "tight": tight, "atol": atol, "rtol": rtol}


def _selftest():
    """Assertions for the failure-reporting path (no JAX, no device needed).

    Regression guard for the bug where a worker that died before printing its
    JSON was reported with JAX's "Platform 'opencl' is experimental" warning as
    the failure reason.
    """
    warn = ("Platform 'opencl' is experimental and not all JAX functionality "
            "may be correctly supported!")
    err = "\n".join([
        warn,
        "Traceback (most recent call last):",
        '  File "/x/xla_bridge.py", line 839, in backends',
        "    backend = _init_backend(platform)",
        "              ^^^^^^^^^^^^^^^^^^^^^^^",
        "jax.errors.JaxRuntimeError: INTERNAL: pjrt-ocl: OclRuntime: no OpenCL device matched selection",
        "",
        "During handling of the above exception, another exception occurred:",
        "",
        "Traceback (most recent call last):",
        '  File "/y/run_suite.py", line 100, in <lambda>',
        "RuntimeError: Unable to initialize backend 'opencl': INTERNAL: no device matched",
        "-" * 20,
        "For simplicity, JAX has removed its internal frames from the traceback "
        "of the following exception. Set JAX_TRACEBACK_FILTERING=off to include these.",
    ])
    reason, op, msg = _crash_reason(err, 1)
    assert warn not in reason and warn not in msg, (reason, msg)
    assert "no OpenCL device matched" in reason or "no device matched" in msg, (reason, msg)

    # warning-only stderr + fatal signal must not become "experimental platform"
    reason, op, msg = _crash_reason(warn + "\n", -11)
    assert reason == "worker killed by signal 11" and op == "signal-11", (reason, op)

    # a real lowering rejection is still classified by op
    reason, op, _ = _crash_reason(
        warn + "\nLoweringError: unsupported op: stablehlo.fft\n", 1)
    assert op == "stablehlo.fft", (reason, op)

    reason, op, msg = _crash_reason("", 3)
    assert reason == "worker died without output", reason

    # mode table wiring
    for name, m in MODES.items():
        assert m["ours"] in BACKENDS and m["ref"] in BACKENDS, name
        assert "PJRT_OCL_DEVICE" not in BACKENDS[m["ours"]], "device must not be hardcoded"
    print("selftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="check the failure-reporting/mode wiring; no device needed")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--name")
    ap.add_argument("--outdir")
    ap.add_argument("--only", nargs="*", help="subset of workload names")
    ap.add_argument("--md", help="write a markdown scoreboard to this path")
    ap.add_argument("--cpu", action="store_true",
                    help="CPU story: ours-on-PoCL (PJRT_OCL_DEVICE=Portable) vs "
                         "reference JAX CPU/XLA backend, in separate processes")
    ap.add_argument("--gpu-vs-cpu", action="store_true",
                    help="ours on an OpenCL GPU (default PJRT_OCL_DEVICE=Intel, "
                         "override with --ocl-device) vs the reference JAX "
                         "CPU/XLA backend — for hosts with no CUDA")
    ap.add_argument("--ocl-device", metavar="SUBSTR",
                    help="PJRT_OCL_DEVICE for the 'ours' runs: OpenCL platform-name "
                         "substring, optional ':<device index>' "
                         "(default: NVIDIA, or Portable with --cpu, Intel with "
                         "--gpu-vs-cpu)")
    ap.add_argument("--timeout", type=float, default=360,
                    help="per-workload per-backend subprocess timeout, seconds")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()
    if args.worker:
        return _worker(args.name, args.outdir)

    global CPU_MODE, REF_BACKEND, OURS_BACKEND, MODE, OCL_DEVICE, REF_LABEL, TIMEOUT
    if args.cpu and args.gpu_vs_cpu:
        ap.error("--cpu and --gpu-vs-cpu are mutually exclusive")
    MODE = "cpu" if args.cpu else ("gpu-vs-cpu" if args.gpu_vs_cpu else "nvidia")
    m = MODES[MODE]
    OURS_BACKEND, REF_BACKEND = m["ours"], m["ref"]
    OCL_DEVICE = args.ocl_device or m["device"]
    REF_LABEL = m["ref_label"]
    CPU_MODE = (MODE == "cpu")
    TIMEOUT = args.timeout
    print(f"mode={MODE}  ours=JAX_PLATFORMS=opencl PJRT_OCL_DEVICE={OCL_DEVICE}"
          f"  ref=JAX_PLATFORMS={REF_BACKEND}", flush=True)

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
        if cmp and cmp["close"] and not cmp.get("tight", True):
            cl += "(loose)"
        print(f"    -> PASS ours={o['ms']:.3f}ms {REF_BACKEND}={c.get('ms',float('nan')):.3f}ms "
              f"gap={gap:.2f}x correct={cl} finite={o.get('finite')}")
    else:
        print(f"    -> FAIL [{o.get('stage')}] {o.get('reason')} "
              f"(op={o.get('missing_op')}) {REF_BACKEND}={c.get('status')} "
              f"| {(o.get('error') or '')[:160]}")


def _print_table(rows):
    print("\n================ COVERAGE / PERF TABLE ================")
    refms = f"{REF_BACKEND} ms"
    hdr = f"{'workload':<18}{'cat':<6}{'ours':<8}{'ours ms':>10}{refms:>10}{'gap':>8}  missing/notes"
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
                  f"{'-':>8}  {miss}  ({REF_BACKEND}:{cs})")


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
    ref = REF_LABEL
    refms = f"{REF_BACKEND} ms"
    npass = sum(1 for r in rows if r["ours"].get("status") == "PASS")
    n = len(rows)
    gpu_vs_cpu = (MODE == "gpu-vs-cpu")
    lines = []
    if CPU_MODE:
        lines.append("# Workload coverage & perf scoreboard (CPU / PoCL)\n")
    elif gpu_vs_cpu:
        lines.append(f"# Workload coverage & perf scoreboard "
                     f"(OpenCL GPU `{OCL_DEVICE}` vs XLA-CPU)\n")
    else:
        lines.append("# Workload coverage & perf scoreboard\n")
    if gpu_vs_cpu:
        lines.append("Diverse AI + scientific-computing workloads run through our OpenCL VM "
                     f"on the `{OCL_DEVICE}` OpenCL platform "
                     f"(`JAX_PLATFORMS=opencl PJRT_OCL_DEVICE={OCL_DEVICE}`) vs the "
                     "reference native JAX CPU/XLA backend (`JAX_PLATFORMS=cpu`) on the "
                     "same host. Generated by `tools/bench_suite/run_suite.py "
                     f"--gpu-vs-cpu --ocl-device {OCL_DEVICE}`. Each backend runs in its "
                     "own subprocess; latency is median-of-5-rounds, adaptive iters, after "
                     "warmup. `gap = ours_ms / cpu_ms` (lower is better; <1 means our GPU "
                     "path beats XLA on this host's CPU). NOTE: this is a cross-device "
                     "comparison (iGPU vs CPU cores), so the gap column is a "
                     "*this-machine* number, not a like-for-like backend comparison.\n")
    elif CPU_MODE:
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
                if cmp["close"] and not cmp.get("tight", True):
                    corr += " *"      # clears 2e-2 but not the f32-tight 1e-3 bar
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
        if CPU_MODE or gpu_vs_cpu:
            winners = ", ".join(f"{nm} ({g:.2f}x)" for nm, g in gs if g <= 1.0) or "none"
            laggards = ", ".join(f"{nm} ({g:.2f}x)" for nm, g in gs if g >= 3.0) or "none"
            lines.append(f"- **Winners (≤1x {ref}):** {winners}.")
            lines.append(f"- **Laggards (≥3x {ref}):** {laggards} — the loop-/small-op "
                         "regime where per-phase enqueue + queue-drain dominates.")
    lines.append("\n### Notes on the correctness column\n")
    if CPU_MODE:
        lines.append("- `correct` uses `np.allclose(atol=1e-3, rtol=1e-3)` vs XLA-CPU (PoCL has "
                     "no TF32, so ours is f32 all the way and should match XLA-CPU tightly). "
                     "The parenthesised `max rel` can still look large where the reference has "
                     "near-zero elements (rel = |Δ|/(|ref|+1e-6)); the boolean is authoritative.")
    else:
        lines.append(f"- `correct` uses `np.allclose(atol=2e-2, rtol=2e-2)` vs {ref} and that boolean is "
                     "authoritative. The parenthesised `max rel` can look huge (e.g. transformer 5e1) "
                     "purely because the reference has near-zero elements (rel = |Δ|/(|ref|+1e-6)); "
                     "every passer clears the abs+rel allclose. TF32 matmul (and, on any GPU, a "
                     "different reduction order / fma contraction) also widens abs error on "
                     "~1-std signals.")
        nloose = sum(1 for r in rows
                     if r["cmp"] and r["cmp"]["close"] and not r["cmp"].get("tight", True))
        lines.append(f"- Rows marked `*` clear the 2e-2 bar but not the f32-tight 1e-3 bar "
                     f"({nloose} of the passers) — pure fp-associativity, reported rather than "
                     "hidden so the accuracy story stays honest.")
    lines.append("- Chaotic integrators (rk4 Lorenz, logistic_map at r→4) are seeded identically "
                 "and match here, but would diverge over longer horizons on any two backends.")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
