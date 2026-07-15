"""Lane-timeline plots: the schedule the compiler INTENDED vs what the device
actually DID (including bubbles).

Top panel   — scheduler intent: the per-lane entry streams produced by
              pjrt_ocl.scheduler, laid out on the cost model's clock (every
              tile of a tile-op class costs `unit_cost` there, so block width
              = tiles x unit cost). Vertical lines are the global barriers the
              scheduler places between dataflow levels.
Bottom panel — measured reality: the same program executed through the plugin
              with PJRT_OCL_VM_TRACE=<file>, which runs the host-dispatch
              engine with one single-workgroup launch per schedule entry on a
              per-lane profiling queue and records per-entry device timestamps.
              White gaps are bubbles: a lane waiting at a barrier for slower
              lanes (imbalance), or lanes idle because a level has fewer tiles
              than lanes.

Because the trace mode launches one kernel per entry it adds launch overhead
(~10-30 us per entry on PoCL) — read the bottom panel as a timeline, not a
benchmark (tools/plot_bench.py benchmarks the untraced engines).

Usage:
    . ./env.sh
    .venv/bin/python tools/plot_schedule.py                      # diamond example
    .venv/bin/python tools/plot_schedule.py --example chain --device Portable
    .venv/bin/python tools/plot_schedule.py --stablehlo prog.mlir  # planned only
    # options: --lanes 8  --device Portable  --cost-table costs.json
    #          --out tools/schedule_plot.png

The measured run needs the plugin .so (PJRT_OCL_PLUGIN_PATH or the dev build
tree) and an OpenCL device (--device selects by platform substring).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "python"))   # prefer this checkout's pjrt_ocl

# --- example programs --------------------------------------------------------
# Each builds (jitted_fn, args). `diamond` is the interesting one: a matmul and
# an elementwise chain that are independent (parallel work for different lanes)
# followed by ops that need both (sequential joins).


def build_example(name: str):
    import jax
    import jax.numpy as jnp
    import numpy as np

    rng = np.random.default_rng(0)

    def m256():
        return jnp.asarray(rng.standard_normal((256, 256)).astype(np.float32)
                           * 0.1)

    if name == "diamond":
        def f(a, b, c):
            m = a @ b        # heavy matmul            \  same dataflow level:
            s = c + c        # cheap elementwise        } independent -> run
            p = c * c        # cheap elementwise       /  in parallel (lanes)
            q = s * p        # needs s,p  -> next level (after the barrier)
            return q + m     # needs q,m  -> final level (the join)
        return jax.jit(f), (m256(), m256(), m256())
    if name == "chain":
        def f(c):            # strictly sequential: one op per level
            x = c + c
            x = x * c
            x = x + x
            x = x * x
            return x + c
        return jax.jit(f), (m256(),)
    if name == "wide":
        def f(c, d):         # eight independent ops in one level, then a
            parts = [c + d, c * d, c - d, jnp.maximum(c, d),   # sequential
                     jnp.minimum(c, d), c * c, d * d, c + c]   # add-funnel
            s = parts[0]
            for pt in parts[1:]:
                s = s + pt
            return s
        return jax.jit(f), (m256(), m256())
    raise SystemExit(f"unknown --example {name!r} (have: diamond, chain, wide)")


# --- artifact + schedule (compile-time view, no OpenCL involved) --------------


def artifact_of(fn, args) -> bytes:
    from jaxlib.mlir.dialects import stablehlo
    art = fn.lower(*args).compiler_ir("stablehlo")
    with art.context:
        return stablehlo.serialize_portable_artifact(
            art, stablehlo.get_current_version())


def artifact_from_file(path: pathlib.Path) -> bytes:
    """A VHLO portable artifact as-is, or textual StableHLO (serialized here)."""
    raw = path.read_bytes()
    from pjrt_ocl import lowering as L
    try:
        L.deserialize_artifact(raw)
        return raw
    except Exception:
        pass
    from jaxlib.mlir import ir
    from jaxlib.mlir.dialects import stablehlo
    with ir.Context() as ctx:
        stablehlo.register_dialect(ctx)
        module = ir.Module.parse(raw.decode())
        return stablehlo.serialize_portable_artifact(
            module, stablehlo.get_current_version())


def plan_schedule(artifact: bytes, lanes: int, cost_table: str | None):
    from pjrt_ocl import lowering as L
    from pjrt_ocl import scheduler as S
    costs = {}
    if cost_table:
        costs = {k: float(v) for k, v in
                 json.loads(pathlib.Path(cost_table).read_text()).items()}
    config = S.DeviceConfig(nlanes=lanes, costs=costs)
    prog = L.lower_artifact(artifact)
    return S, config, S.schedule_program(prog, config)


# --- op naming (block labels + fixed color order) -----------------------------

TILE_NAMES = {0: "ew", 1: "matmul", 2: "gather", 3: "reduce", 4: "reduce-comb",
              5: "iota", 6: "scatter", 7: "dyn-slice", 8: "dyn-update",
              9: "reduce-win"}
EW_SUB_NAMES = {0: "add", 1: "mul", 2: "sub", 3: "div", 4: "max", 5: "min",
                6: "pow", 7: "copy", 8: "neg", 9: "exp", 10: "log", 11: "sqrt",
                12: "rsqrt", 13: "tanh", 14: "abs", 15: "floor", 16: "ceil",
                17: "sign", 18: "fill", 20: "cmp", 21: "select", 23: "convert",
                24: "bitcast", 25: "atan2", 26: "rem"}


def task_label(tile_op: int, p0: int) -> str:
    base = tile_op & 0xFF
    if base == 0:
        return EW_SUB_NAMES.get(p0, f"ew#{p0}")
    return TILE_NAMES.get(base, f"op#{base}")


# --- planned timeline ---------------------------------------------------------
# Replays the per-lane streams on the cost model's clock. A BARRIER synchronizes
# every lane to the max; entry duration = tiles x unit_cost. Root level only
# (a WHILE's runtime iteration count is unknowable at schedule time).


def planned_timeline(S, config, sched):
    n = sched.n_lanes
    root = (sched.root_lens if sched.root_lens is not None
            else [len(s) for s in sched.lane_streams])
    pos, t = [0] * n, [0.0] * n
    blocks, barriers = [], []       # (lane, start, dur, task_id), [x...]
    while True:
        for lane in range(n):
            stream = sched.lane_streams[lane]
            while pos[lane] < root[lane]:
                e = stream[pos[lane]]
                if e.task == S.TASK_BARRIER:
                    break
                if e.task in (S.TASK_WHILE, S.TASK_IF):
                    raise SystemExit(
                        "planned timeline: control flow (while/if) has no "
                        "static duration; use an example without loops")
                if e.task != S.TASK_NOP:
                    task = sched.tasks[e.task]
                    dur = ((e.tile_hi - e.tile_lo)
                           * config.unit_cost(task.tile_op))
                    blocks.append((lane, t[lane], dur, e.task))
                    t[lane] += dur
                pos[lane] += 1
        if all(pos[lane] >= root[lane] for lane in range(n)):
            break
        tmax = max(t)
        barriers.append(tmax)
        t = [tmax] * n
        for lane in range(n):       # step past the barrier entry
            if pos[lane] < root[lane]:
                pos[lane] += 1
    return blocks, barriers, max(t) if t else 0.0


# --- measured timeline (trace file) -------------------------------------------


def read_trace(trace_path: pathlib.Path, want_tasks, lanes: int):
    """Last trace line whose task table matches the expected schedule (other
    lines belong to warmup-compiled helper programs jax runs eagerly)."""
    want = [(t.tile_op & 0xFF, t.p0, t.p1) for t in want_tasks]
    match = None
    for line in trace_path.read_text().splitlines():
        rec = json.loads(line)
        got = [(t[0] & 0xFF, t[2], t[3]) for t in rec["tasks"]]
        if rec["n_lanes"] == lanes and got == want:
            match = rec
    if match is None:
        raise SystemExit("trace: no line matches the expected schedule "
                         f"(looked for {len(want)} tasks, {lanes} lanes)")
    return match


def measured_timeline(rec):
    evs = rec["events"]
    if not evs:
        raise SystemExit("trace: no profiled events in the matching line")
    t0 = min(e[6] for e in evs)
    blocks = [(e[1], (e[6] - t0) / 1e3, (e[7] - e[6]) / 1e3, e[3])
              for e in evs]                       # lane, start us, dur us, task
    phase_end = {}
    for e in evs:
        phase_end[e[0]] = max(phase_end.get(e[0], 0.0), (e[7] - t0) / 1e3)
    barriers = [v for _, v in sorted(phase_end.items())[:-1]]
    makespan = max(e[7] for e in evs) - t0
    return blocks, barriers, makespan / 1e3


# --- plotting ------------------------------------------------------------------
# Colors: validated categorical palette (dataviz skill), fixed op->slot order of
# first appearance. Low-contrast slots get relief via direct labels + edges.

PALETTE = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834",
           "#4a3aa7", "#e34948"]
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"


def draw_panel(ax, blocks, barriers, makespan, n_lanes, color_of, label_of,
               xlabel, title, bubble_note=None):
    for lane, start, dur, task in blocks:
        ax.barh(lane, dur, left=start, height=0.62,
                color=color_of(task), edgecolor=SURFACE, linewidth=0.8,
                zorder=3)
        if dur > makespan * 0.045:  # direct label when the block can carry it
            ax.text(start + dur / 2, lane, label_of(task), ha="center",
                    va="center", fontsize=7.5, color=SURFACE, zorder=4,
                    fontweight="bold")
    for x in barriers:
        ax.axvline(x, color=INK2, lw=1.0, ls=(0, (4, 3)), zorder=2, alpha=0.8)
    if bubble_note:
        for lane, pct in bubble_note.items():
            ax.text(makespan * 1.01, lane, f"{pct:.0f}% idle", va="center",
                    fontsize=7.5, color=INK2)
    ax.set_xlim(0, makespan * (1.12 if bubble_note else 1.02))
    ax.set_ylim(n_lanes - 0.4, -0.6)
    ax.set_yticks(range(n_lanes),
                  [f"lane {i}" for i in range(n_lanes)], fontsize=8)
    ax.set_xlabel(xlabel, fontsize=9, color=INK)
    ax.set_title(title, fontsize=10, loc="left", color=INK)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, axis="x", ls=":", alpha=0.35, zorder=0)
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_visible(False)


def plot(sched, planned, measured, device, name, out: pathlib.Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # fixed color order: op label by first appearance in the planned stream
    labels = {}
    for t in sched.tasks:
        labels.setdefault(task_label(t.tile_op, t.p0), len(labels))
    slot = {lab: PALETTE[i % len(PALETTE)] if i < len(PALETTE) else "#9a9a94"
            for lab, i in labels.items()}
    label_of = lambda tid: task_label(sched.tasks[tid].tile_op,
                                      sched.tasks[tid].p0)
    color_of = lambda tid: slot[label_of(tid)]

    n = sched.n_lanes
    nrows = 2 if measured else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 1.1 + 0.42 * n * nrows),
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)

    blocks, barriers, makespan = planned
    draw_panel(axes[0][0], blocks, barriers, makespan, n, color_of, label_of,
               "cost-model time (unit-cost x tiles)",
               f"{name}: scheduled — per-lane streams on the cost model's "
               "clock (dashed = global barrier)")

    if measured:
        blocks, barriers, makespan = measured
        busy = {lane: 0.0 for lane in range(n)}
        for lane, _s, dur, _t in blocks:
            busy[lane] += dur
        bubbles = {lane: 100.0 * (1 - busy[lane] / makespan)
                   for lane in range(n)}
        total = 100.0 * (1 - sum(busy.values()) / (makespan * n))
        draw_panel(axes[1][0], blocks, barriers, makespan, n, color_of,
                   label_of, "measured time (µs, device clock)",
                   f"measured on {device} — white gaps are bubbles "
                   f"({total:.0f}% of lane-time idle)", bubble_note=bubbles)

    fig.legend(handles=[Patch(facecolor=c, label=l) for l, c in slot.items()],
               loc="upper right", ncol=min(len(slot), 6), fontsize=8,
               frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"wrote {out}")


# --- worker: run the example through the plugin with tracing on ---------------


def run_worker(example: str) -> int:
    import jax
    fn, args = build_example(example)
    for _ in range(2):                     # compile + warm caches
        jax.block_until_ready(fn(*args))
    jax.block_until_ready(fn(*args))       # the traced run we plot (last line)
    return 0


def collect_trace(example: str, device: str, lanes: int,
                  cost_table: str | None) -> pathlib.Path:
    trace = pathlib.Path(tempfile.mkstemp(suffix=".trace.jsonl")[1])
    env = dict(os.environ)
    env.update(JAX_PLATFORMS="opencl", PJRT_OCL_DEVICE=device,
               PJRT_OCL_VM_LANES=str(lanes), PJRT_OCL_VM_TRACE=str(trace))
    env.setdefault("PJRT_OCL_PLUGIN_PATH",
                   str(REPO / "build" / "pjrt_plugin" / "libpjrt_ocl.so"))
    env.setdefault("POCL_CACHE_DIR", str(REPO / "third_party" / "pocl-cache"))
    if cost_table:
        env["PJRT_OCL_COST_TABLE"] = str(pathlib.Path(cost_table).resolve())
    proc = subprocess.run(
        [sys.executable, str(pathlib.Path(__file__)), "--worker", example],
        env=env, timeout=600, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"traced run failed (rc={proc.returncode})")
    return trace


# --- driver --------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Plot scheduled vs measured per-lane execution timelines.")
    ap.add_argument("--example", default="diamond",
                    help="built-in program: diamond | chain | wide")
    ap.add_argument("--stablehlo", metavar="FILE",
                    help="plot the planned timeline of this StableHLO "
                         "(VHLO artifact or .mlir text); skips the measured run")
    ap.add_argument("--device", default=os.environ.get("PJRT_OCL_DEVICE",
                                                       "Portable"))
    ap.add_argument("--lanes", type=int, default=8)
    ap.add_argument("--cost-table", default=os.environ.get(
        "PJRT_OCL_COST_TABLE") or None,
        help="JSON {ew_tile_us, mma_tile_us, ...} for the cost-model clock")
    ap.add_argument("--out", default=str(REPO / "tools" / "schedule_plot.png"))
    ap.add_argument("--worker", metavar="EXAMPLE", help="internal")
    args = ap.parse_args()

    if args.worker:
        raise SystemExit(run_worker(args.worker))

    # The driver only lowers/schedules — pin jax to CPU so importing jax does
    # NOT route eager ops through the plugin (docs/decisions.md #4).
    os.environ["JAX_PLATFORMS"] = "cpu"

    if args.stablehlo:
        artifact = artifact_from_file(pathlib.Path(args.stablehlo))
    else:
        fn, fargs = build_example(args.example)
        artifact = artifact_of(fn, fargs)

    S, config, sched = plan_schedule(artifact, args.lanes, args.cost_table)
    planned = planned_timeline(S, config, sched)

    measured = None
    if not args.stablehlo:
        trace = collect_trace(args.example, args.device, args.lanes,
                              args.cost_table)
        rec = read_trace(trace, sched.tasks, args.lanes)
        measured = measured_timeline(rec)
        device = rec["device"]
        trace.unlink()
    else:
        device = args.device
        print("note: --stablehlo plots the planned timeline only "
              "(no host program to execute)")

    name = (pathlib.Path(args.stablehlo).name if args.stablehlo
            else args.example)
    plot(sched, planned, measured, device, name, pathlib.Path(args.out))


if __name__ == "__main__":
    main()
